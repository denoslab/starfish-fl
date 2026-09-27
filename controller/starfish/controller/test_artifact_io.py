"""Tests for safe artifact serialization (artifact_io) and a guard that keeps
unsafe deserialization out of the controller code.

See babelbrain-docs/specs/starfish-work-items.md, SF-01.
"""

import ast
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np

from starfish.controller.file.artifact_io import (
    META_KEY,
    REQUIRED_META,
    SCHEMA,
    ArtifactError,
    load_artifact,
    save_artifact,
    sha256_file,
    tensors_to_weights,
    validate_meta,
    weights_to_tensors,
)

CONTROLLER_PKG = Path(__file__).resolve().parent

META = {
    'task': 'TestTask',
    'model_version': 'p1-b1-t1-r1',
    'n_samples': 12,
    'round': 1,
    'metrics': {'loss': 0.25, 'per_class': [1, 2]},
}


def meta(**overrides):
    out = dict(META)
    out.update(overrides)
    return out


class SaveLoadRoundTripTest(TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, 'nested', 'dir', 'artifact')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dtypes_and_shapes_round_trip(self):
        rng = np.random.default_rng(0)
        tensors = {
            'f16': rng.standard_normal((4, 3)).astype(np.float16),
            'f32': rng.standard_normal((2, 3, 5)).astype(np.float32),
            'f64': rng.standard_normal(7),
            'i64': np.arange(10, dtype=np.int64),
            'u8': np.arange(6, dtype=np.uint8).reshape(2, 3),
            'bool': np.array([True, False, True]),
            'scalar': np.array(3.5, dtype=np.float32),
            'empty': np.zeros((0, 4), dtype=np.float32),
        }
        save_artifact(self.path, tensors, META)
        loaded, _ = load_artifact(self.path)
        self.assertEqual(set(loaded), set(tensors))
        for name, arr in tensors.items():
            self.assertEqual(loaded[name].dtype, arr.dtype, name)
            self.assertEqual(loaded[name].shape, arr.shape, name)
            np.testing.assert_array_equal(loaded[name], arr)

    def test_non_contiguous_array_is_stored_correctly(self):
        arr = np.arange(12.0).reshape(3, 4).T
        self.assertFalse(arr.flags.c_contiguous)
        save_artifact(self.path, {'t': arr}, META)
        loaded, _ = load_artifact(self.path)
        np.testing.assert_array_equal(loaded['t'], arr)

    def test_metadata_round_trip_adds_schema(self):
        save_artifact(self.path, {'w': np.ones(2)}, META)
        _, loaded_meta = load_artifact(self.path)
        self.assertEqual(loaded_meta['schema'], SCHEMA)
        for key, value in META.items():
            self.assertEqual(loaded_meta[key], value)

    def test_delta_metadata_round_trip(self):
        delta_meta = meta(kind='delta', base_version='250k-v0002')
        save_artifact(self.path, {'w': np.ones(2)}, delta_meta)
        _, loaded_meta = load_artifact(self.path)
        self.assertEqual(loaded_meta['base_version'], '250k-v0002')

    def test_caller_meta_is_not_modified(self):
        original = dict(META)
        save_artifact(self.path, {'w': np.ones(2)}, original)
        self.assertEqual(original, META)

    def test_metadata_only_artifact(self):
        save_artifact(self.path, {}, meta(note='no tensors'))
        tensors, loaded_meta = load_artifact(self.path)
        self.assertEqual(tensors, {})
        self.assertEqual(loaded_meta['note'], 'no tensors')

    def test_write_is_atomic_and_leaves_no_temp_files(self):
        save_artifact(self.path, {'w': np.ones(3)}, META)
        folder = os.path.dirname(self.path)
        self.assertEqual(os.listdir(folder), ['artifact'])

    def test_overwrite_replaces_previous_file(self):
        save_artifact(self.path, {'w': np.ones(3)}, META)
        save_artifact(self.path, {'w': np.zeros(3)}, META)
        loaded, _ = load_artifact(self.path)
        np.testing.assert_array_equal(loaded['w'], np.zeros(3))


class IntegrityTest(TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, 'artifact')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returned_hash_matches_file(self):
        sha = save_artifact(self.path, {'w': np.arange(5.0)}, META)
        self.assertEqual(sha, sha256_file(self.path))
        self.assertEqual(len(sha), 64)

    def test_load_with_correct_hash(self):
        sha = save_artifact(self.path, {'w': np.arange(5.0)}, META)
        loaded, _ = load_artifact(self.path, expected_sha256=sha.upper())
        np.testing.assert_array_equal(loaded['w'], np.arange(5.0))

    def test_tampered_file_fails_hash_check(self):
        sha = save_artifact(self.path, {'w': np.arange(64.0)}, META)
        with open(self.path, 'r+b') as f:
            f.seek(-8, os.SEEK_END)
            f.write(b'\x00' * 8)
        with self.assertRaisesRegex(ArtifactError, 'SHA-256 mismatch'):
            load_artifact(self.path, expected_sha256=sha)

    def test_corrupt_header_raises_artifact_error(self):
        with open(self.path, 'wb') as f:
            f.write(b'\xff' * 64)
        with self.assertRaises(ArtifactError):
            load_artifact(self.path)

    def test_truncated_file_raises_artifact_error(self):
        save_artifact(self.path, {'w': np.arange(64.0)}, META)
        size = os.path.getsize(self.path)
        with open(self.path, 'r+b') as f:
            f.truncate(size - 16)
        with self.assertRaises(ArtifactError):
            load_artifact(self.path)

    def test_missing_file_raises_artifact_error(self):
        with self.assertRaises(ArtifactError):
            load_artifact(os.path.join(self.tmp, 'nope'))

    def test_pickle_file_is_rejected_without_executing(self):
        marker = Path(self.tmp) / 'pwned'

        class Exploit:
            def __reduce__(self):
                return (open, (str(marker), 'w'))

        with open(self.path, 'wb') as f:
            pickle.dump({'weights': Exploit()}, f)
        with self.assertRaises(ArtifactError):
            load_artifact(self.path)
        self.assertFalse(marker.exists())


class InputValidationTest(TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, 'artifact')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_object_dtype_rejected(self):
        with self.assertRaises(ArtifactError):
            save_artifact(
                self.path, {'bad': np.array([{'a': 1}], dtype=object)}, META)
        self.assertFalse(os.path.exists(self.path))

    def test_non_dict_tensors_rejected(self):
        with self.assertRaises(ArtifactError):
            save_artifact(self.path, [np.ones(2)], META)

    def test_non_serialisable_metadata_rejected(self):
        with self.assertRaises(ArtifactError):
            save_artifact(self.path, {'w': np.ones(2)}, meta(bad=object()))
        self.assertFalse(os.path.exists(self.path))

    def test_each_required_key_is_enforced_on_save(self):
        for key in REQUIRED_META:
            if key == 'schema':
                continue
            incomplete = dict(META)
            del incomplete[key]
            with self.assertRaisesRegex(ArtifactError, key, msg=key):
                save_artifact(self.path, {'w': np.ones(2)}, incomplete)
        self.assertFalse(os.path.exists(self.path))

    def test_delta_requires_base_version(self):
        with self.assertRaisesRegex(ArtifactError, 'base_version'):
            save_artifact(self.path, {'w': np.ones(2)}, meta(kind='delta'))

    def test_bad_metadata_types_rejected(self):
        for bad in (meta(n_samples=-1), meta(n_samples=1.5), meta(n_samples=True),
                    meta(round='2'), meta(metrics=[0.1]), meta(schema='other/9')):
            with self.assertRaises(ArtifactError, msg=bad):
                validate_meta(dict(bad, schema=bad.get('schema', SCHEMA)))

    def test_foreign_safetensors_file_needs_strict_false(self):
        from safetensors.numpy import save_file
        save_file({'w': np.ones(2, dtype=np.float32)}, self.path)
        with self.assertRaisesRegex(ArtifactError, META_KEY):
            load_artifact(self.path)
        tensors, loaded_meta = load_artifact(self.path, strict=False)
        self.assertEqual(loaded_meta, {})
        np.testing.assert_array_equal(
            tensors['w'], np.ones(2, dtype=np.float32))

    def test_incomplete_metadata_rejected_on_load(self):
        from safetensors.numpy import save_file
        save_file({'w': np.ones(2)}, self.path, metadata={
            META_KEY: json.dumps({'schema': SCHEMA, 'task': 'X'})})
        with self.assertRaisesRegex(ArtifactError, 'missing'):
            load_artifact(self.path)
        _, loaded_meta = load_artifact(self.path, strict=False)
        self.assertEqual(loaded_meta['task'], 'X')

    def test_unknown_schema_rejected_on_load(self):
        from safetensors.numpy import save_file
        save_file({'w': np.ones(2)}, self.path, metadata={
            META_KEY: json.dumps(dict(META, schema='starfish-artifact/99'))})
        with self.assertRaisesRegex(ArtifactError, 'schema'):
            load_artifact(self.path)

    def test_invalid_metadata_json_rejected(self):
        from safetensors.numpy import save_file
        save_file({'w': np.ones(2)}, self.path, metadata={
                  META_KEY: '{not json'})
        with self.assertRaises(ArtifactError):
            load_artifact(self.path, strict=False)

    def test_non_object_metadata_json_rejected(self):
        from safetensors.numpy import save_file
        save_file({'w': np.ones(2)}, self.path, metadata={META_KEY: '[1, 2]'})
        with self.assertRaises(ArtifactError):
            load_artifact(self.path, strict=False)


class WeightListTest(TestCase):

    def test_order_preserved_past_ten_items(self):
        weights = [np.full(2, i, dtype=np.float32) for i in range(12)]
        tensors = weights_to_tensors(weights)
        self.assertIn('w0011', tensors)
        restored = tensors_to_weights(dict(reversed(list(tensors.items()))))
        for i, w in enumerate(restored):
            np.testing.assert_array_equal(w, weights[i])

    def test_gap_in_indices_rejected(self):
        with self.assertRaises(ArtifactError):
            tensors_to_weights({'w0000': np.ones(1), 'w0002': np.ones(1)})

    def test_unexpected_name_rejected(self):
        with self.assertRaises(ArtifactError):
            tensors_to_weights({'w0000': np.ones(1), 'bias': np.ones(1)})


UNSAFE_MODULES = {'pickle', 'cPickle', '_pickle', 'dill', 'cloudpickle',
                  'joblib', 'shelve', 'marshal'}


def _is_false(node):
    return isinstance(node, ast.Constant) and node.value is False


def _is_true(node):
    return isinstance(node, ast.Constant) and node.value is True


def find_unsafe_deserialization(source, filename='<source>'):
    """Return a list of unsafe deserialization sites found in ``source``.

    Flags imports of pickle-like modules, any ``allow_pickle`` argument that
    is not a literal False, ``read_pickle`` calls, and ``torch.load`` calls
    without ``weights_only=True``.
    """
    problems = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in UNSAFE_MODULES:
                    problems.append('{}:{} imports {}'.format(
                        filename, node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if (node.module or '').split('.')[0] in UNSAFE_MODULES:
                problems.append('{}:{} imports from {}'.format(
                    filename, node.lineno, node.module))
        elif isinstance(node, ast.Call):
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            if 'allow_pickle' in kwargs and not _is_false(kwargs['allow_pickle']):
                problems.append('{}:{} passes allow_pickle'.format(
                    filename, node.lineno))
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr == 'read_pickle':
                    problems.append('{}:{} calls read_pickle'.format(
                        filename, node.lineno))
                if (func.attr == 'load' and isinstance(func.value, ast.Name)
                        and func.value.id == 'torch'
                        and not _is_true(kwargs.get('weights_only'))):
                    problems.append('{}:{} calls torch.load without weights_only=True'.format(
                        filename, node.lineno))
    return problems


class NoUnsafeDeserializationTest(TestCase):
    """Fails if controller code outside tests can deserialize with pickle.

    Data from other sites arrives through the router. Loading it with pickle,
    or with numpy's allow_pickle, would let a malicious site run code on the
    coordinator. Use starfish.controller.file.artifact_io instead.
    """

    def _source_files(self):
        for path in CONTROLLER_PKG.rglob('*.py'):
            rel = path.relative_to(CONTROLLER_PKG)
            if path.name.startswith('test_') or path.name == 'tests.py' or 'tests' in rel.parts:
                continue
            if 'migrations' in rel.parts:
                continue
            yield path

    def test_scan_covers_the_task_code(self):
        scanned = {p.relative_to(CONTROLLER_PKG).as_posix()
                   for p in self._source_files()}
        self.assertIn('tasks/federated_unet/task.py', scanned)
        self.assertIn('file/artifact_io.py', scanned)

    def test_no_unsafe_deserialization_in_controller_code(self):
        problems = []
        for path in self._source_files():
            problems.extend(find_unsafe_deserialization(
                path.read_text(), str(path)))
        self.assertEqual(
            problems, [], 'Unsafe deserialization found:\n' + '\n'.join(problems))

    def test_checker_flags_known_unsafe_patterns(self):
        unsafe = [
            'import pickle\npickle.load(f)',
            'import pickle as p\np.loads(b)',
            'from pickle import loads',
            'import _pickle',
            'import dill',
            'import joblib',
            'from joblib import load',
            'import numpy as np\nnp.load(f, allow_pickle=True)',
            'import numpy as np\nnp.load(f, allow_pickle=flag)',
            'import pandas as pd\npd.read_pickle(f)',
            'import torch\ntorch.load(f)',
            'import torch\ntorch.load(f, weights_only=False)',
        ]
        for snippet in unsafe:
            self.assertTrue(find_unsafe_deserialization(snippet), snippet)

    def test_checker_allows_safe_patterns(self):
        safe = [
            'import numpy as np\nnp.load(f)',
            'import numpy as np\nnp.load(f, allow_pickle=False)',
            'import torch\ntorch.load(f, weights_only=True)',
            'import json\njson.load(f)',
            'from safetensors.numpy import load_file',
        ]
        for snippet in safe:
            self.assertEqual(find_unsafe_deserialization(snippet), [], snippet)
