"""Integration tests for FederatedUNet with actual TensorFlow on CPU.

These tests are NOT run in the standard CI (no TF installed). They run
in the separate unet-tests.yml workflow which installs the unet group.

Uses tiny 4x4 images with 2 samples and 1 epoch to keep runtime under
a minute even on CPU.
"""

import io
import os
import pickle
import shutil
import tempfile
import zipfile

import numpy as np
import pytest

# Skip entire module if TensorFlow is not available
tf = pytest.importorskip('tensorflow')

from starfish.controller.tasks.federated_unet.task import FederatedUNet
from starfish.controller.file.file_utils import load_image_dataset_by_run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PATCH_SIZE = 4
N_IMAGES = 2


@pytest.fixture(autouse=True)
def tmp_base_folder(tmp_path, monkeypatch):
    """Redirect all file_utils I/O to a temp directory."""
    monkeypatch.setattr(
        'starfish.controller.file.file_utils.base_folder', str(tmp_path))
    return tmp_path


def make_run(cur_seq=1, current_round=1, total_round=1, role='coordinator'):
    return {
        'id': 42, 'project': 7, 'batch': 1, 'role': role,
        'status': 'standby', 'cur_seq': cur_seq,
        'tasks': [{'config': {
            'current_round': current_round,
            'total_round': total_round,
            'patch_size': PATCH_SIZE,
            'architecture': 'mobilenetv2',
            'type_Unet': 'unet',
            'local_epochs': 1,
            'batch_size': N_IMAGES,
            'learning_rate': 0.001,
        }}],
    }


def write_dataset_zip(base_dir, run_id=42):
    """Write a tiny dataset zip for testing."""
    from PIL import Image

    dataset_dir = os.path.join(str(base_dir), str(run_id))
    os.makedirs(dataset_dir, exist_ok=True)
    zip_path = os.path.join(dataset_dir, 'dataset')

    rng = np.random.default_rng(42)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for i in range(N_IMAGES):
            name = f'{i:03d}.png'
            img = Image.fromarray(
                (rng.random((PATCH_SIZE, PATCH_SIZE)) * 255).astype(np.uint8),
                mode='L')
            mask = Image.fromarray(
                ((rng.random((PATCH_SIZE, PATCH_SIZE)) > 0.5).astype(np.uint8) * 255),
                mode='L')

            for prefix, im in [('images', img), ('masks', mask)]:
                b = io.BytesIO()
                im.save(b, format='PNG')
                zf.writestr(f'{prefix}/{name}', b.getvalue())

    with open(zip_path, 'wb') as f:
        f.write(buf.getvalue())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildModel:
    """Verify model construction works on CPU with a lightweight backbone."""

    def test_build_returns_keras_model(self):
        task = FederatedUNet(make_run())
        task._set_runtime_env()
        model = task._build_model()
        assert hasattr(model, 'get_weights')
        assert hasattr(model, 'fit')

    def test_model_output_shape(self):
        task = FederatedUNet(make_run())
        task._set_runtime_env()
        model = task._build_model()
        dummy = tf.zeros((1, PATCH_SIZE, PATCH_SIZE, 1))
        output = model(dummy)
        assert output.shape == (1, PATCH_SIZE, PATCH_SIZE, 1)


class TestPrepareData:
    """Verify data loading and model initialization."""

    def test_prepare_data_succeeds(self, tmp_base_folder):
        write_dataset_zip(tmp_base_folder)
        task = FederatedUNet(make_run())
        assert task.prepare_data() is True
        assert task.sample_size == N_IMAGES
        assert task.model is not None

    def test_prepare_data_fails_without_dataset(self, tmp_base_folder):
        task = FederatedUNet(make_run())
        assert task.prepare_data() is False


class TestTraining:
    """Verify a single training step produces valid output."""

    def test_training_produces_mid_artifact(self, tmp_base_folder):
        write_dataset_zip(tmp_base_folder)
        task = FederatedUNet(make_run())
        task.prepare_data()
        assert task.training() is True

        mid_path = os.path.join(
            str(tmp_base_folder), '42', '1', '1', 'mid-artifacts')
        assert os.path.exists(mid_path)

        with open(mid_path, 'rb') as f:
            result = pickle.load(f)

        assert 'weights' in result
        assert 'n_samples' in result
        assert result['n_samples'] == N_IMAGES
        assert 'metrics' in result
        assert result['metrics']['loss'] >= 0

    def test_training_metrics_valid_ranges(self, tmp_base_folder):
        write_dataset_zip(tmp_base_folder)
        task = FederatedUNet(make_run())
        task.prepare_data()
        task.training()

        mid_path = os.path.join(
            str(tmp_base_folder), '42', '1', '1', 'mid-artifacts')
        with open(mid_path, 'rb') as f:
            result = pickle.load(f)

        metrics = result['metrics']
        assert 0 <= metrics['iou_score'] <= 1
        assert 0 <= metrics['f1-score'] <= 1


class TestEndToEnd:
    """Full prepare -> train -> aggregate cycle on CPU."""

    def test_full_cycle(self, tmp_base_folder):
        write_dataset_zip(tmp_base_folder)
        task = FederatedUNet(make_run())

        # Prepare + train
        assert task.prepare_data() is True
        assert task.training() is True

        # Simulate a second site's mid-artifact
        mid_path = os.path.join(
            str(tmp_base_folder), '42', '1', '1', 'mid-artifacts')
        with open(mid_path, 'rb') as f:
            site1_result = pickle.load(f)

        # Write both as mid-artifacts for aggregation
        agg_dir = os.path.join(
            str(tmp_base_folder), 'all-mid-artifacts', '7', '1')
        os.makedirs(agg_dir, exist_ok=True)

        for site_name in ['siteA', 'siteB']:
            with open(os.path.join(agg_dir, f'{site_name}-1-1-mid-artifacts'), 'wb') as f:
                pickle.dump(site1_result, f)

        # Aggregate (mock upload since there's no router)
        from unittest.mock import patch as mock_patch
        with mock_patch.object(FederatedUNet, 'upload', return_value=True):
            assert task.do_aggregate() is True

        # Verify aggregated artifact exists and is loadable
        artifact_path = os.path.join(
            str(tmp_base_folder), '42', '1', '1', 'artifacts')
        assert os.path.exists(artifact_path)

        with open(artifact_path, 'rb') as f:
            agg_weights = pickle.load(f)

        # Aggregated weights should be a list of numpy arrays
        assert isinstance(agg_weights, list)
        assert all(isinstance(w, np.ndarray) for w in agg_weights)

        # Since both sites have identical weights, aggregation = original
        for orig, agg in zip(site1_result['weights'], agg_weights):
            np.testing.assert_allclose(agg, orig, atol=1e-6)
