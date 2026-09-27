"""Model weights as named numpy arrays, deltas and FedAvg, for BabelBrainFno.

Weights travel through SF-01 safetensors artifacts as float arrays. Complex
parameters, as in FNO spectral layers, are stored as real arrays with a
trailing axis of 2 and listed in ``complex_keys``, so any safetensors
version can hold them. Only ``state_to_arrays`` and ``arrays_to_state``
need torch; the rest is plain numpy.
"""

import hashlib

import numpy as np


class WeightsError(Exception):
    """Raised when weights or deltas do not fit together."""


def state_to_arrays(state):
    """A torch state dict to ``(arrays, complex_keys)``, float arrays on the CPU."""
    import torch
    arrays, complex_keys = {}, []
    for name, tensor in state.items():
        t = tensor.detach().cpu()
        if t.is_complex():
            t = torch.view_as_real(t.contiguous())
            complex_keys.append(name)
        arrays[name] = t.numpy().copy()
    return arrays, sorted(complex_keys)


def arrays_to_state(arrays, complex_keys=()):
    """Inverse of :func:`state_to_arrays`: a state dict of torch tensors."""
    import torch
    complex_keys = set(complex_keys)
    state = {}
    for name, arr in arrays.items():
        t = torch.from_numpy(np.ascontiguousarray(arr))
        state[name] = torch.view_as_complex(t) if name in complex_keys else t
    return state


def check_compatible(reference, other, what, check_dtype=True):
    """Same names, shapes and dtypes, or a WeightsError naming the first difference."""
    if set(reference) != set(other):
        missing = sorted(set(reference) ^ set(other))[:3]
        raise WeightsError(
            '{} has different tensor names, for example {}'.format(what, missing))
    for name, ref in reference.items():
        arr = other[name]
        if arr.shape != ref.shape or (check_dtype and arr.dtype != ref.dtype):
            raise WeightsError('{} tensor {} is {} {}, expected {} {}'.format(
                what, name, arr.dtype, arr.shape, ref.dtype, ref.shape))


def delta(local, base):
    """Local weights minus the global weights they started from."""
    check_compatible(base, local, 'local model')
    return {name: local[name] - base[name] for name in base}


def apply_delta(base, update):
    """``base`` plus ``update``, summed in float64 and cast back to each tensor's dtype."""
    check_compatible(base, update, 'delta', check_dtype=False)
    return {name: (base[name].astype(np.float64) + update[name]).astype(base[name].dtype)
            for name in base}


def fedavg(base, updates):
    """Sample-weighted mean of the deltas, added to ``base``.

    ``updates`` is a list of ``(delta_arrays, n_samples)``. The mean is taken
    in float64 and cast back to each tensor's dtype.
    """
    if not updates:
        raise WeightsError('no updates to aggregate')
    total = sum(n for _, n in updates)
    if total <= 0:
        raise WeightsError('updates carry no samples')
    for i, (update, n) in enumerate(updates):
        check_compatible(base, update, 'update {}'.format(i))
        for name, arr in update.items():
            if not np.all(np.isfinite(arr)):
                raise WeightsError(
                    'update {} has non-finite values in {}'.format(i, name))
    mean = {name: sum(update[name].astype(np.float64) * (n / total) for update, n in updates)
            for name in base}
    return apply_delta(base, mean)


def digest(arrays):
    """SHA-256 over names, shapes, dtypes and bytes, independent of dict order."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        arr = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def update_norm(update):
    """L2 norm of a whole delta, over every tensor."""
    return float(np.sqrt(sum(float(np.sum(np.square(v, dtype=np.float64))) for v in update.values())))


def screen(updates, clip_norm=None, screen_factor=None):
    """Robust aggregation, SF-11: drop bad deltas, then clip the rest.

    ``updates`` is a list of ``(delta, n_samples)``. Returns ``(kept, excluded)``:
    ``kept`` holds ``(index, delta, n_samples)`` with clipping applied, and
    ``excluded`` holds ``(index, reason)``. A delta with non-finite values is
    always excluded. With ``screen_factor``, a delta whose norm is more than
    that many times the median norm is excluded; this needs at least three
    deltas, since the median of two cannot tell which one is off. With
    ``clip_norm``, a kept delta longer than that is scaled down to it.
    """
    candidates, excluded = [], []
    for index, (update, n) in enumerate(updates):
        if not all(np.all(np.isfinite(v)) for v in update.values()):
            excluded.append((index, 'non-finite values'))
            continue
        candidates.append((index, update, n, update_norm(update)))
    if screen_factor and len(candidates) >= 3:
        median = float(np.median([c[3] for c in candidates]))
        if median > 0:
            limit = float(screen_factor) * median
            for c in list(candidates):
                if c[3] > limit:
                    candidates.remove(c)
                    excluded.append((c[0], 'norm {:.4g} is more than {} times the median {:.4g}'.format(
                        c[3], screen_factor, median)))
    kept = []
    for index, update, n, norm in candidates:
        if clip_norm and norm > float(clip_norm):
            factor = float(clip_norm) / norm
            update = {name: (v * factor).astype(v.dtype)
                      for name, v in update.items()}
        kept.append((index, update, n))
    return kept, sorted(excluded)
