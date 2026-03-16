"""Unit tests for FederatedUNet task.

Tests the pure-logic parts (FedAvg aggregation, artifact serialization,
image format validation) without requiring TensorFlow. The model build/train
paths are mocked so these tests run in the standard CI.

Tests that require Pillow (dataset loading) are skipped when Pillow is not
installed (it lives in the optional unet dependency group).
"""

import io
import json
import os
import pickle
import shutil
import tempfile
import zipfile

import numpy as np
from django.test import TestCase
from pathlib import Path
from unittest import skipUnless
from unittest.mock import patch, MagicMock

from starfish.controller.tasks.federated_unet.task import FederatedUNet
from starfish.controller.file.file_utils import _is_supported_image

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run(role='coordinator', cur_seq=1, current_round=1, total_round=3):
    return {
        'id': 42, 'project': 7, 'batch': 1, 'role': role,
        'status': 'standby', 'cur_seq': cur_seq,
        'tasks': [{'config': {
            'current_round': current_round,
            'total_round': total_round,
            'patch_size': 8,
            'local_epochs': 1,
            'batch_size': 1,
            'learning_rate': 0.001,
        }}],
    }


def make_dataset_zip(n_images=4, size=8):
    """Create an in-memory zip with tiny grayscale images and masks.

    Requires Pillow to generate PNG files.
    """
    buf = io.BytesIO()
    rng = np.random.default_rng(42)
    with zipfile.ZipFile(buf, 'w') as zf:
        for i in range(n_images):
            name = f'{i:03d}.png'
            img_arr = (rng.random((size, size)) * 255).astype(np.uint8)
            mask_arr = (rng.random((size, size)) > 0.5).astype(np.uint8) * 255

            img = Image.fromarray(img_arr, mode='L')
            mask = Image.fromarray(mask_arr, mode='L')

            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            zf.writestr(f'images/{name}', img_bytes.getvalue())

            mask_bytes = io.BytesIO()
            mask.save(mask_bytes, format='PNG')
            zf.writestr(f'masks/{name}', mask_bytes.getvalue())

    return buf.getvalue()


# ---------------------------------------------------------------------------
# _is_supported_image
# ---------------------------------------------------------------------------

class SupportedImageTest(TestCase):

    def test_png(self):
        self.assertTrue(_is_supported_image('img.png'))

    def test_jpg(self):
        self.assertTrue(_is_supported_image('photo.JPG'))

    def test_tiff(self):
        self.assertTrue(_is_supported_image('scan.tiff'))

    def test_csv_not_supported(self):
        self.assertFalse(_is_supported_image('data.csv'))

    def test_no_extension(self):
        self.assertFalse(_is_supported_image('README'))


# ---------------------------------------------------------------------------
# load_image_dataset_by_run
# ---------------------------------------------------------------------------

@skipUnless(HAS_PILLOW, 'Pillow not installed (unet group)')
class LoadImageDatasetTest(TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()
        from starfish.controller.file.file_utils import load_image_dataset_by_run
        self.load_image_dataset_by_run = load_image_dataset_by_run

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_dataset_zip(self, run_id=42, n_images=4, size=8):
        dataset_dir = os.path.join(self.tmp_dir, str(run_id))
        os.makedirs(dataset_dir, exist_ok=True)
        zip_path = os.path.join(dataset_dir, 'dataset')
        with open(zip_path, 'wb') as f:
            f.write(make_dataset_zip(n_images=n_images, size=size))

    def test_loads_correct_shape(self):
        self._write_dataset_zip(n_images=4, size=8)
        images, masks = self.load_image_dataset_by_run(42, patch_size=8)
        self.assertEqual(images.shape, (4, 8, 8))
        self.assertEqual(masks.shape, (4, 8, 8))

    def test_images_normalized_0_1(self):
        self._write_dataset_zip()
        images, _ = self.load_image_dataset_by_run(42, patch_size=8)
        self.assertGreaterEqual(images.min(), 0.0)
        self.assertLessEqual(images.max(), 1.0)

    def test_masks_are_binary(self):
        self._write_dataset_zip()
        _, masks = self.load_image_dataset_by_run(42, patch_size=8)
        unique_vals = set(np.unique(masks))
        self.assertTrue(unique_vals.issubset({0.0, 1.0}))

    def test_resize_to_patch_size(self):
        self._write_dataset_zip(size=16)
        images, masks = self.load_image_dataset_by_run(42, patch_size=4)
        self.assertEqual(images.shape[1:], (4, 4))
        self.assertEqual(masks.shape[1:], (4, 4))

    def test_returns_none_when_zip_missing(self):
        images, masks = self.load_image_dataset_by_run(999, patch_size=8)
        self.assertIsNone(images)
        self.assertIsNone(masks)

    def test_returns_none_for_empty_zip(self):
        dataset_dir = os.path.join(self.tmp_dir, '42')
        os.makedirs(dataset_dir, exist_ok=True)
        zip_path = os.path.join(dataset_dir, 'dataset')
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w'):
            pass  # empty zip
        with open(zip_path, 'wb') as f:
            f.write(buf.getvalue())
        images, masks = self.load_image_dataset_by_run(42, patch_size=8)
        self.assertIsNone(images)

    def test_returns_none_for_count_mismatch(self):
        """Zip with 2 images but 1 mask should fail."""
        dataset_dir = os.path.join(self.tmp_dir, '42')
        os.makedirs(dataset_dir, exist_ok=True)
        zip_path = os.path.join(dataset_dir, 'dataset')

        from PIL import Image
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            for name in ['001.png', '002.png']:
                img = Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode='L')
                b = io.BytesIO()
                img.save(b, format='PNG')
                zf.writestr(f'images/{name}', b.getvalue())
            # Only one mask
            img = Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode='L')
            b = io.BytesIO()
            img.save(b, format='PNG')
            zf.writestr('masks/001.png', b.getvalue())

        with open(zip_path, 'wb') as f:
            f.write(buf.getvalue())
        images, masks = self.load_image_dataset_by_run(42, patch_size=8)
        self.assertIsNone(images)


# ---------------------------------------------------------------------------
# FedAvg aggregation (do_aggregate)
# ---------------------------------------------------------------------------

class FedAvgAggregationTest(TestCase):
    """Test FedAvg weighted averaging logic without TensorFlow."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_task(self, **kwargs):
        return FederatedUNet(make_run(**kwargs))

    def _write_mid_artifact(self, filename, weights, n_samples):
        """Write a pickled mid-artifact with model weights and sample count."""
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)
        payload = {
            'weights': weights,
            'n_samples': n_samples,
            'metrics': {'loss': 0.5, 'iou_score': 0.6, 'f1-score': 0.7},
        }
        with open(dir_path / filename, 'wb') as f:
            pickle.dump(payload, f)

    @patch.object(FederatedUNet, 'upload', return_value=True)
    def test_equal_weight_averaging(self, _):
        """Two sites with equal samples → simple average of weights."""
        w1 = [np.array([1.0, 2.0]), np.array([3.0])]
        w2 = [np.array([3.0, 4.0]), np.array([5.0])]
        self._write_mid_artifact('siteA-1-1-mid-artifacts', w1, 100)
        self._write_mid_artifact('siteB-1-1-mid-artifacts', w2, 100)

        task = self._make_task()
        self.assertTrue(task.do_aggregate())

        artifact_path = os.path.join(self.tmp_dir, '42', '1', '1', 'artifacts')
        with open(artifact_path, 'rb') as f:
            result = pickle.load(f)

        np.testing.assert_allclose(result[0], [2.0, 3.0])
        np.testing.assert_allclose(result[1], [4.0])

    @patch.object(FederatedUNet, 'upload', return_value=True)
    def test_sample_weighted_averaging(self, _):
        """Site with 3x samples gets 3x the weight."""
        w1 = [np.array([0.0, 0.0])]
        w2 = [np.array([4.0, 8.0])]
        self._write_mid_artifact('siteA-1-1-mid-artifacts', w1, 25)
        self._write_mid_artifact('siteB-1-1-mid-artifacts', w2, 75)

        task = self._make_task()
        task.do_aggregate()

        artifact_path = os.path.join(self.tmp_dir, '42', '1', '1', 'artifacts')
        with open(artifact_path, 'rb') as f:
            result = pickle.load(f)

        # 25/100 * [0,0] + 75/100 * [4,8] = [3.0, 6.0]
        np.testing.assert_allclose(result[0], [3.0, 6.0])

    @patch.object(FederatedUNet, 'upload', return_value=True)
    def test_single_site_returns_own_weights(self, _):
        w = [np.array([1.0, 2.0, 3.0])]
        self._write_mid_artifact('site1-1-1-mid-artifacts', w, 50)

        task = self._make_task()
        task.do_aggregate()

        artifact_path = os.path.join(self.tmp_dir, '42', '1', '1', 'artifacts')
        with open(artifact_path, 'rb') as f:
            result = pickle.load(f)

        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0])

    def test_returns_false_when_no_mid_artifacts(self):
        (Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1').mkdir(
            parents=True, exist_ok=True)
        task = self._make_task()
        self.assertFalse(task.do_aggregate())

    @patch.object(FederatedUNet, 'upload', return_value=True)
    def test_aggregate_calls_upload(self, mock_upload):
        w = [np.array([1.0])]
        self._write_mid_artifact('site1-1-1-mid-artifacts', w, 50)
        task = self._make_task()
        task.do_aggregate()
        mock_upload.assert_called_once_with(True)


# ---------------------------------------------------------------------------
# Artifact save/load round-trip
# ---------------------------------------------------------------------------

class ArtifactSerializationTest(TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_and_load_round_trip(self):
        task = FederatedUNet(make_run())
        weights = [np.random.randn(3, 3), np.random.randn(5)]
        url = os.path.join(self.tmp_dir, '42', '1', '1', 'artifacts')

        self.assertTrue(task.save_artifacts(url, weights))
        self.assertTrue(os.path.exists(url))

        with open(url, 'rb') as f:
            loaded = pickle.load(f)

        self.assertEqual(len(loaded), 2)
        np.testing.assert_array_equal(loaded[0], weights[0])
        np.testing.assert_array_equal(loaded[1], weights[1])

    def test_save_returns_false_for_none(self):
        task = FederatedUNet(make_run())
        self.assertFalse(task.save_artifacts('/tmp/doesnt_matter', None))


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

class ValidateTest(TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @patch.object(FederatedUNet, 'is_first_round', return_value=True)
    def test_first_round_always_valid(self, _):
        task = FederatedUNet(make_run())
        self.assertTrue(task.validate())

    @patch.object(FederatedUNet, 'is_first_round', return_value=False)
    @patch.object(FederatedUNet, 'download_artifact', return_value=False)
    def test_later_round_fails_without_artifact(self, *_):
        task = FederatedUNet(make_run(current_round=2))
        self.assertFalse(task.validate())
