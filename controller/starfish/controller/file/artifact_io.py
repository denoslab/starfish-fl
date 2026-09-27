"""Safe serialization for model artifacts exchanged between sites.

Artifacts are stored as safetensors files. Unlike pickle, reading a
safetensors file can never execute code, so a coordinator can open files
sent by any participant. Metadata such as sample counts and metrics travels
inside the file header as a JSON string under the key ``starfish_meta``.

Every Starfish artifact carries the metadata keys in ``REQUIRED_META``, and
a delta, ``kind == 'delta'``, also carries ``base_version``, the global model
version it applies to. See babelbrain-docs/specs/starfish-work-items.md, SF-01.

Typical use::

    sha = save_artifact(path, weights_to_tensors(model_weights), {
        'task': 'FederatedUNet', 'model_version': 'p7-b1-t1-r2',
        'n_samples': 120, 'round': 2, 'metrics': {'loss': 0.3}})
    tensors, meta = load_artifact(path, expected_sha256=sha)
    model_weights = tensors_to_weights(tensors)
"""

import hashlib
import json
import os
import re
import tempfile

import numpy as np

META_KEY = 'starfish_meta'
SCHEMA = 'starfish-artifact/1'
REQUIRED_META = ('schema', 'task', 'model_version',
                 'n_samples', 'round', 'metrics')
DELTA_KIND = 'delta'
_WEIGHT_KEY = re.compile(r'^w(\d{4,})$')


class ArtifactError(Exception):
    """Raised when an artifact cannot be written, verified or read."""


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_meta(meta):
    """Raise :class:`ArtifactError` unless ``meta`` is complete Starfish metadata."""
    if not isinstance(meta, dict):
        raise ArtifactError('metadata must be a dict')
    required = list(REQUIRED_META)
    if meta.get('kind') == DELTA_KIND:
        required.append('base_version')
    missing = [key for key in required if key not in meta]
    if missing:
        raise ArtifactError(
            'metadata is missing {}'.format(', '.join(missing)))
    if meta['schema'] != SCHEMA:
        raise ArtifactError('unsupported artifact schema {!r}, expected {!r}'.format(
            meta['schema'], SCHEMA))
    if not _is_int(meta['n_samples']) or meta['n_samples'] < 0:
        raise ArtifactError('n_samples must be a non-negative integer')
    if not _is_int(meta['round']):
        raise ArtifactError('round must be an integer')
    if not isinstance(meta['metrics'], dict):
        raise ArtifactError('metrics must be a dict')


def sha256_file(path, chunk_size=1 << 20):
    """Return the hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk_size), b''):
            digest.update(block)
    return digest.hexdigest()


def save_artifact(path, tensors, meta):
    """Write ``tensors`` and ``meta`` to ``path`` as one safetensors file.

    Parameters
    ----------
    path : str
        Destination file. Parent folders are created. The write is atomic:
        a reader never sees a half-written file.
    tensors : dict[str, numpy.ndarray]
        Named arrays. Object arrays are rejected.
    meta : dict
        JSON-serialisable metadata with every key in ``REQUIRED_META``
        except ``schema``, which is added automatically.

    Returns
    -------
    str
        Hex SHA-256 of the written file.

    Raises
    ------
    ArtifactError
        If a tensor or the metadata cannot be stored safely, or the metadata
        is incomplete. Nothing is written in that case.
    """
    from safetensors.numpy import save_file

    if not isinstance(tensors, dict):
        raise ArtifactError('tensors must be a dict of name to array')
    arrays = {}
    for name, value in tensors.items():
        if not isinstance(name, str) or not name:
            raise ArtifactError('tensor names must be non-empty strings')
        arr = np.asarray(value)
        if arr.dtype == object:
            raise ArtifactError(
                'tensor {} has object dtype, which cannot be stored safely'.format(name))
        # np.ascontiguousarray would turn 0-d arrays into 1-d ones, so copy only when needed.
        arrays[name] = arr if arr.flags.c_contiguous else np.array(
            arr, order='C')

    if not isinstance(meta, dict):
        raise ArtifactError('metadata must be a dict')
    meta = dict(meta)
    meta.setdefault('schema', SCHEMA)
    validate_meta(meta)
    try:
        meta_json = json.dumps(meta)
    except (TypeError, ValueError) as e:
        raise ArtifactError('metadata is not JSON serialisable: {}'.format(e))

    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=folder, prefix='.tmp-', suffix='.safetensors')
    os.close(fd)
    try:
        save_file(arrays, tmp_path, metadata={META_KEY: meta_json})
        os.replace(tmp_path, path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if isinstance(e, ArtifactError):
            raise
        raise ArtifactError('could not write artifact {}: {}'.format(path, e))
    return sha256_file(path)


def load_artifact(path, expected_sha256=None, strict=True):
    """Read an artifact written by :func:`save_artifact`.

    Parameters
    ----------
    path : str
        File to read.
    expected_sha256 : str, optional
        When given, the file hash must match before anything is parsed.
    strict : bool, default True
        Require complete Starfish metadata, see :func:`validate_meta`. Pass
        False only to read a safetensors file from another producer, such as
        converted seed weights. Its metadata is then ``{}`` when absent.

    Returns
    -------
    tuple[dict[str, numpy.ndarray], dict]
        The arrays and the metadata dict.

    Raises
    ------
    ArtifactError
        If the file is missing, fails the hash check, is not a valid
        safetensors file, or has invalid metadata. A pickle file is rejected
        here without being run.
    """
    from safetensors import safe_open

    if not os.path.isfile(path):
        raise ArtifactError('artifact not found: {}'.format(path))
    if expected_sha256 is not None:
        actual = sha256_file(path)
        if actual.lower() != str(expected_sha256).lower():
            raise ArtifactError('SHA-256 mismatch for {}: expected {}, got {}'.format(
                path, expected_sha256, actual))
    try:
        with safe_open(path, framework='np') as f:
            header_meta = f.metadata() or {}
            tensors = {name: f.get_tensor(name) for name in f.keys()}
    except Exception as e:
        raise ArtifactError(
            '{} is not a valid safetensors artifact: {}'.format(path, e))

    raw = header_meta.get(META_KEY)
    if raw is None:
        if strict:
            raise ArtifactError(
                '{} has no {} metadata'.format(path, META_KEY))
        return tensors, {}
    try:
        meta = json.loads(raw)
    except ValueError as e:
        raise ArtifactError(
            'metadata in {} is not valid JSON: {}'.format(path, e))
    if not isinstance(meta, dict):
        raise ArtifactError('metadata in {} is not a JSON object'.format(path))
    if strict:
        try:
            validate_meta(meta)
        except ArtifactError as e:
            raise ArtifactError('invalid metadata in {}: {}'.format(path, e))
    return tensors, meta


def weights_to_tensors(weights):
    """Map an ordered list of arrays, such as Keras ``get_weights()``, to named tensors."""
    return {'w{:04d}'.format(i): np.asarray(w) for i, w in enumerate(weights)}


def tensors_to_weights(tensors):
    """Inverse of :func:`weights_to_tensors`. Keys must be w0000, w0001, ... without gaps."""
    indexed = []
    for name, value in tensors.items():
        match = _WEIGHT_KEY.match(name)
        if not match:
            raise ArtifactError(
                'unexpected tensor name {} in a weight list artifact'.format(name))
        indexed.append((int(match.group(1)), value))
    indexed.sort(key=lambda item: item[0])
    if [i for i, _ in indexed] != list(range(len(indexed))):
        raise ArtifactError(
            'weight list artifact has missing or duplicate indices')
    return [value for _, value in indexed]
