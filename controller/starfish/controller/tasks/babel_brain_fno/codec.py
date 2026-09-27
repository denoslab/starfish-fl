"""Compression of weight deltas for BabelBrainFno, SF-07.

``encode`` turns a delta into the tensors that travel; ``decode`` turns them
back into a full delta on the coordinator. The method and its parameters go
into the artifact metadata under ``codec``, so each delta is decoded the way
its site encoded it. Task config key ``compression``, default ``none``:

- ``{"method": "fp16"}``: values in float16, about half the size.
- ``{"method": "int8"}``: symmetric int8 per tensor with a float32 scale, about a quarter.
- ``{"method": "topk", "k": 0.01}``: per tensor, only the largest share ``k``
  of entries by magnitude, as float16 values and int32 indices, with error
  feedback: what a site does not send is kept locally and added to its next
  round's delta. About 65 times smaller at 1%.

A tensor left out of the encoded delta, as in partial fine-tuning, decodes as
zeros. Each method must be measured against uncompressed FL in E1 before it
is switched on for real runs.
"""

import math

import numpy as np

METHODS = ('none', 'fp16', 'int8', 'topk')
DEFAULT_TOPK = 0.01
_SEP = '::'


class CodecError(Exception):
    """An encoded delta that cannot be decoded safely."""


def config(cfg):
    cfg = dict(cfg or {})
    method = cfg.get('method', 'none')
    if method not in METHODS:
        raise CodecError(
            'compression method must be one of {}'.format(', '.join(METHODS)))
    if method == 'topk':
        k = float(cfg.get('k', DEFAULT_TOPK))
        if not 0 < k <= 1:
            raise CodecError('topk k must be in (0, 1]')
        cfg['k'] = k
    cfg['method'] = method
    return cfg


def encode(delta, cfg, residual=None):
    """``(tensors, codec_meta, new_residual)``. ``new_residual`` is None except for topk."""
    cfg = config(cfg)
    method = cfg['method']
    tensors = {}
    if method == 'none':
        tensors = {name: np.asarray(v, dtype=np.float32)
                   for name, v in delta.items()}
        return tensors, {'method': method}, None
    if method == 'fp16':
        tensors = {name: np.asarray(v).astype(np.float16)
                   for name, v in delta.items()}
        return tensors, {'method': method}, None
    if method == 'int8':
        for name, v in delta.items():
            v = np.asarray(v, dtype=np.float32)
            peak = float(np.abs(v).max()) if v.size else 0.0
            scale = peak / 127.0 if peak > 0 else 1.0
            tensors[name + _SEP +
                    'q'] = np.clip(np.rint(v / scale), -127, 127).astype(np.int8)
            tensors[name + _SEP +
                    'scale'] = np.array([scale], dtype=np.float32)
        return tensors, {'method': method}, None

    # topk with error feedback
    new_residual = {}
    for name, v in delta.items():
        x = np.asarray(v, dtype=np.float32)
        if residual is not None and name in residual and residual[name].shape == x.shape:
            x = x + residual[name]
        flat = x.ravel()
        k = max(1, int(math.ceil(cfg['k'] * flat.size))) if flat.size else 0
        index_dtype = np.int32 if flat.size < 2 ** 31 else np.int64
        idx = np.argpartition(np.abs(flat), flat.size -
                              k)[flat.size - k:] if k else np.array([], int)
        idx = np.sort(idx).astype(index_dtype)
        values = flat[idx].astype(np.float16)
        tensors[name + _SEP + 'idx'] = idx
        tensors[name + _SEP + 'val'] = values
        sent = np.zeros_like(flat)
        sent[idx] = values.astype(np.float32)
        new_residual[name] = (flat - sent).reshape(x.shape)
    return tensors, {'method': 'topk', 'k': cfg['k']}, new_residual


def decode(tensors, codec_meta, base):
    """The full delta, float32, one array per tensor of ``base``; missing tensors are zeros.

    Non-finite values pass through; robust aggregation, SF-11, excludes such deltas.
    """
    method = (codec_meta or {}).get('method', 'none')
    if method not in METHODS:
        raise CodecError('unknown codec {!r}'.format(method))
    out = {name: np.zeros(arr.shape, dtype=np.float32)
           for name, arr in base.items()}

    def target(name):
        if name not in base:
            raise CodecError(
                'delta has tensor {} that the model does not have'.format(name))
        return base[name]

    if method in ('none', 'fp16'):
        for name, v in tensors.items():
            ref = target(name)
            if v.shape != ref.shape:
                raise CodecError('tensor {} has shape {}, expected {}'.format(
                    name, v.shape, ref.shape))
            out[name] = v.astype(np.float32)
    else:
        groups = {}
        for key, v in tensors.items():
            name, sep, part = key.rpartition(_SEP)
            if not sep:
                raise CodecError(
                    'encoded tensor {} has no part suffix'.format(key))
            groups.setdefault(name, {})[part] = v
        for name, parts in groups.items():
            ref = target(name)
            if method == 'int8':
                if set(parts) != {'q', 'scale'} or parts['q'].shape != ref.shape:
                    raise CodecError(
                        'int8 tensor {} is malformed'.format(name))
                scale = float(parts['scale'].reshape(-1)[0])
                if not np.isfinite(scale) or scale <= 0:
                    raise CodecError(
                        'int8 tensor {} has a bad scale'.format(name))
                out[name] = parts['q'].astype(np.float32) * scale
            else:
                if set(parts) != {'idx', 'val'} or parts['idx'].shape != parts['val'].shape:
                    raise CodecError(
                        'topk tensor {} is malformed'.format(name))
                idx = parts['idx'].astype(np.int64)
                if idx.size and (idx.min() < 0 or idx.max() >= ref.size):
                    raise CodecError(
                        'topk tensor {} has indices out of range'.format(name))
                if len(np.unique(idx)) != idx.size:
                    raise CodecError(
                        'topk tensor {} repeats an index'.format(name))
                flat = np.zeros(ref.size, dtype=np.float32)
                flat[idx] = parts['val'].astype(np.float32)
                out[name] = flat.reshape(ref.shape)
    return out


def nbytes(tensors):
    return int(sum(np.asarray(v).nbytes for v in tensors.values()))
