"""Tests for the BabelBrain sample store reader, SF-03.

All samples are synthetic, written by ``synthetic.py``.
"""

import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from unittest import TestCase, skipUnless
from unittest.mock import patch

import h5py
import numpy as np

from starfish.controller.tasks.babel_brain_fno import store as store_module
from starfish.controller.tasks.babel_brain_fno.sample_io import (
    SampleFormatError, read_sample, torch_dataset)
from starfish.controller.tasks.babel_brain_fno.store import (
    SCHEMA_PATH, STORE_ENV, SampleStore, StoreError)
from starfish.controller.tasks.babel_brain_fno.synthetic import (
    append_manifest, delete_sample, write_sample_file, write_synthetic_store)

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def flip_bytes(path, n=16):
    """Corrupt a file in the middle, where HDF5 keeps data, not padding."""
    size = os.path.getsize(path)
    with open(path, 'r+b') as f:
        f.seek(size // 2)
        chunk = f.read(n)
        f.seek(size // 2)
        f.write(bytes(b ^ 0xFF for b in chunk))


def find_docs_schema():
    """The manifest schema in a sibling babelbrain-docs checkout, if there is one."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'babelbrain-docs' / 'specs' / 'manifest.schema.json'
        if candidate.is_file():
            return candidate
    return None


class StoreTestCase(TestCase):
    """A fresh synthetic store of 20 samples: 16 train in 4 groups, 4 val in 1 group."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.join(self.tmp, 'store')
        self.entries = write_synthetic_store(
            self.root, n_groups=5, per_group=4, val_groups=1)
        self.logger = logging.getLogger('test-store-{}'.format(id(self)))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def open(self, **kwargs):
        return SampleStore(self.root, logger=self.logger, **kwargs)

    def sample_path(self, entry):
        return os.path.join(self.root, 'v1', entry['file'])

    def ids(self, records):
        return {r.sample_id for r in records}


class CountsTest(StoreTestCase):

    def test_counts_per_split(self):
        store = self.open()
        self.assertEqual(store.counts(250000), {'train': 16, 'val': 4})
        self.assertEqual(len(store.samples(250000, 'train')), 16)
        self.assertEqual(len(store.samples(250000, 'val')), 4)
        self.assertEqual(dict(store.rejections), {})

    def test_split_is_by_group(self):
        store = self.open()
        val_groups = {r.group_id for r in store.samples(split='val')}
        train_groups = {r.group_id for r in store.samples(split='train')}
        self.assertEqual(len(val_groups), 1)
        self.assertEqual(len(train_groups), 4)
        self.assertFalse(val_groups & train_groups)

    def test_filter_by_bucket(self):
        write_synthetic_store(self.root, n_groups=2, per_group=3, val_groups=0,
                              bucket_hz=500000, seed=1)
        store = self.open()
        self.assertEqual(store.counts(500000), {'train': 6, 'val': 0})
        self.assertEqual(store.counts(250000), {'train': 16, 'val': 4})
        self.assertEqual(len(store.samples()), 26)
        self.assertTrue(
            all(r.bucket_hz == 500000 for r in store.samples(500000)))

    def test_records_point_at_files_inside_the_store(self):
        for record in self.open().samples():
            self.assertTrue(record.path.startswith(
                os.path.realpath(self.root)))
            self.assertTrue(os.path.isfile(record.path))

    def test_record_repr_hides_path(self):
        record = self.open().samples()[0]
        self.assertNotIn(record.path, repr(record))


class TombstoneTest(StoreTestCase):

    def test_deleted_sample_is_excluded(self):
        gone = self.entries[5]
        delete_sample(self.root, gone['sample_id'])
        store = self.open()
        self.assertNotIn(gone['sample_id'], self.ids(store.samples()))
        self.assertEqual(store.counts(250000), {'train': 15, 'val': 4})
        self.assertEqual(store.deleted, 1)
        self.assertEqual(dict(store.rejections), {})

    def test_tombstone_wins_even_if_the_file_remains(self):
        gone = self.entries[0]
        append_manifest(
            self.root, {'sample_id': gone['sample_id'], 'deleted': True})
        self.assertTrue(os.path.exists(self.sample_path(gone)))
        self.assertNotIn(gone['sample_id'], self.ids(self.open().samples()))

    def test_tombstone_before_the_sample_line_still_deletes(self):
        manifest = os.path.join(self.root, 'v1', 'manifest.jsonl')
        gone = self.entries[3]
        with open(manifest) as f:
            lines = f.readlines()
        tombstone = json.dumps(
            {'sample_id': gone['sample_id'], 'deleted': True}) + '\n'
        with open(manifest, 'w') as f:
            f.writelines([tombstone] + lines)
        self.assertNotIn(gone['sample_id'], self.ids(self.open().samples()))


class IntegrityTest(StoreTestCase):

    def test_tampered_file_is_excluded_with_a_warning(self):
        bad = self.entries[2]
        flip_bytes(self.sample_path(bad))
        store = self.open()
        with self.assertLogs(self.logger, level='WARNING') as logs:
            records = store.samples()
        self.assertNotIn(bad['sample_id'], self.ids(records))
        self.assertEqual(len(records), 19)
        self.assertEqual(store.rejections['sha256_mismatch'], 1)
        self.assertTrue(any(bad['sample_id'] in m and 'sha256_mismatch' in m
                            for m in logs.output))

    def test_missing_file_is_excluded(self):
        os.remove(self.sample_path(self.entries[0]))
        store = self.open()
        self.assertEqual(len(store.samples()), 19)
        self.assertEqual(store.rejections['missing_file'], 1)

    def test_manifest_path_traversal_is_rejected(self):
        entry = dict(self.entries[0], sample_id=str(uuid.uuid4()),
                     file='../../etc/passwd')
        append_manifest(self.root, entry)
        store = self.open()
        self.assertEqual(len(store.samples()), 20)
        self.assertEqual(store.rejections['schema'], 1)

    def test_symlink_out_of_the_store_is_rejected(self):
        outside = os.path.join(self.tmp, 'outside.h5')
        sample_id = str(uuid.uuid4())
        write_sample_file(outside, sample_id, 250000)
        link = os.path.join(self.root, 'v1', '250000',
                            '{}.h5'.format(sample_id))
        os.symlink(outside, link)
        entry = dict(self.entries[0], sample_id=sample_id,
                     file='250000/{}.h5'.format(sample_id),
                     sha256=store_module._sha256(outside))
        append_manifest(self.root, entry)
        store = self.open()
        self.assertNotIn(sample_id, self.ids(store.samples()))
        self.assertEqual(store.rejections['outside_store'], 1)

    def test_bucket_folder_must_match_bucket(self):
        entry = dict(self.entries[0], sample_id=str(
            uuid.uuid4()), bucket_hz=500000)
        append_manifest(self.root, entry)
        store = self.open()
        self.assertEqual(store.counts(500000), {'train': 0, 'val': 0})
        self.assertEqual(store.rejections['bucket_mismatch'], 1)

    def test_duplicate_sample_line_is_rejected(self):
        append_manifest(self.root, dict(self.entries[0], split='val'))
        store = self.open()
        self.assertEqual(store.counts(250000), {'train': 16, 'val': 4})
        self.assertEqual(store.rejections['duplicate'], 1)

    def test_bad_lines_are_rejected_and_the_rest_used(self):
        manifest = os.path.join(self.root, 'v1', 'manifest.jsonl')
        with open(manifest, 'a') as f:
            f.write('{not json\n\n')
        append_manifest(self.root, dict(self.entries[0], sample_id=str(uuid.uuid4()),
                                        patient_name='should never be here'))
        append_manifest(self.root, dict(
            self.entries[0], sample_id='not-a-uuid'))
        append_manifest(self.root, dict(self.entries[0], sample_id=str(uuid.uuid4()),
                                        ct_type='ZTE'))
        store = self.open()
        self.assertEqual(len(store.samples()), 20)
        self.assertEqual(store.rejections['invalid_json'], 1)
        self.assertEqual(store.rejections['schema'], 3)


class HashCacheTest(StoreTestCase):

    def setUp(self):
        super().setUp()
        self.cache_path = os.path.join(self.tmp, 'controller', 'cache.json')

    def count_hashes(self):
        return patch.object(store_module, '_sha256', wraps=store_module._sha256)

    def test_each_file_is_hashed_once_across_scans(self):
        with self.count_hashes() as spy:
            self.open(cache_path=self.cache_path).scan()
            self.assertEqual(spy.call_count, 20)
            self.open(cache_path=self.cache_path).scan()
            self.assertEqual(spy.call_count, 20)

    def test_changed_file_is_hashed_again_and_rejected(self):
        self.open(cache_path=self.cache_path).scan()
        bad = self.entries[1]
        with open(self.sample_path(bad), 'ab') as f:
            f.write(b'junk')
        store = self.open(cache_path=self.cache_path)
        self.assertNotIn(bad['sample_id'], self.ids(store.samples()))
        self.assertEqual(store.rejections['sha256_mismatch'], 1)

    def test_cache_holds_no_paths(self):
        self.open(cache_path=self.cache_path).scan()
        with open(self.cache_path) as f:
            text = f.read()
        self.assertNotIn(self.tmp, text)
        self.assertNotIn('.h5', text)
        self.assertEqual(len(json.loads(text)), 20)

    def test_corrupt_cache_is_ignored(self):
        os.makedirs(os.path.dirname(self.cache_path))
        with open(self.cache_path, 'w') as f:
            f.write('garbage')
        self.assertEqual(
            len(self.open(cache_path=self.cache_path).samples()), 20)


class PrivacyTest(StoreTestCase):

    def test_logs_never_contain_store_paths(self):
        flip_bytes(self.sample_path(self.entries[0]))
        os.remove(self.sample_path(self.entries[1]))
        append_manifest(self.root, dict(self.entries[2], sample_id=str(uuid.uuid4()),
                                        file='../../etc/passwd'))
        with self.assertLogs(self.logger, level='DEBUG') as logs:
            self.open().scan()
        self.assertTrue(logs.output)
        for message in logs.output:
            self.assertNotIn(self.tmp, message)
            self.assertNotIn(os.path.realpath(self.tmp), message)


class OpenTest(TestCase):

    def test_from_env_requires_the_variable(self):
        with self.assertRaisesRegex(StoreError, STORE_ENV):
            SampleStore.from_env(environ={})

    def test_from_env_uses_the_variable(self):
        tmp = tempfile.mkdtemp()
        try:
            store = SampleStore.from_env(environ={STORE_ENV: tmp})
            self.assertEqual(store.root, os.path.realpath(tmp))
        finally:
            shutil.rmtree(tmp)

    def test_store_without_manifest_raises(self):
        tmp = tempfile.mkdtemp()
        try:
            with self.assertRaises(StoreError) as ctx:
                SampleStore(tmp).samples()
            self.assertNotIn(tmp, str(ctx.exception))
        finally:
            shutil.rmtree(tmp)


class SchemaCopyTest(TestCase):

    def test_vendored_schema_matches_the_docs(self):
        docs_schema = find_docs_schema()
        if docs_schema is None:
            self.skipTest(
                'babelbrain-docs checkout not found next to this repo')
        with open(docs_schema) as f:
            expected = json.load(f)
        with open(SCHEMA_PATH) as f:
            actual = json.load(f)
        self.assertEqual(actual, expected,
                         'Copy babelbrain-docs/specs/manifest.schema.json into {}'.format(
                             SCHEMA_PATH.name))


class ReadSampleTest(StoreTestCase):

    def test_reads_a_synthetic_sample(self):
        arrays, attrs = read_sample(self.sample_path(self.entries[0]))
        self.assertEqual(arrays['ct_hu'].shape, (16, 16, 32))
        self.assertEqual(arrays['skull_field'].shape, (2, 16, 16, 32))
        self.assertEqual(arrays['brain_mask'].dtype, np.uint8)
        self.assertEqual(attrs['frequency_hz'], 250000.0)
        self.assertAlmostEqual(attrs['spacing_mm'], 0.49)

    def _rewrite(self, name, data):
        path = self.sample_path(self.entries[0])
        with h5py.File(path, 'r+') as f:
            del f[name]
            if data is not None:
                f.create_dataset(name, data=data)
        return path

    def test_wrong_dtype_is_rejected(self):
        path = self._rewrite('brain_mask', np.zeros(
            (16, 16, 32), dtype=np.float32))
        with self.assertRaisesRegex(SampleFormatError, 'brain_mask'):
            read_sample(path)

    def test_missing_dataset_is_rejected(self):
        path = self._rewrite('sos', None)
        with self.assertRaisesRegex(SampleFormatError, 'sos'):
            read_sample(path)

    def test_mismatched_grid_is_rejected(self):
        path = self._rewrite('skull_field', np.zeros(
            (2, 16, 16, 30), dtype=np.float32))
        with self.assertRaisesRegex(SampleFormatError, 'skull_field'):
            read_sample(path)

    def test_wrong_schema_version_is_rejected(self):
        path = self.sample_path(self.entries[0])
        with h5py.File(path, 'r+') as f:
            f.attrs['schema_version'] = '2.0'
        with self.assertRaisesRegex(SampleFormatError, 'schema_version'):
            read_sample(path)

    def test_non_hdf5_file_is_rejected(self):
        path = self.sample_path(self.entries[0])
        with open(path, 'wb') as f:
            f.write(b'not hdf5')
        with self.assertRaises(SampleFormatError):
            read_sample(path)

    @skipUnless(HAS_TORCH, 'torch is not installed')
    def test_torch_dataset(self):
        records = self.open().samples(250000, 'train')
        dataset = torch_dataset(records)
        self.assertEqual(len(dataset), 16)
        item = dataset[0]
        self.assertEqual(tuple(item['water_field'].shape), (2, 16, 16, 32))
        self.assertEqual(item['frequency_hz'], 250000.0)
