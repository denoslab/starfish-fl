"""Read one BabelBrain sample file and wrap a store as a torch Dataset.

``read_sample`` needs h5py only. ``torch_dataset`` imports torch lazily, so
this module can be imported on a site without the torch group.
"""

import numpy as np

from starfish.controller.tasks.babel_brain_fno.store import SCHEMA_VERSION

# Dataset name -> (dtype, rank), from the sample contract v1.
SAMPLE_DATASETS = {
    'ct_hu': (np.float32, 3),
    'water_field': (np.float32, 4),
    'skull_field': (np.float32, 4),
    'sos': (np.float32, 3),
    'attenuation': (np.float32, 3),
    'brain_mask': (np.uint8, 3),
}
COMPLEX_FIELDS = ('water_field', 'skull_field')


class SampleFormatError(Exception):
    """Raised when a sample file does not follow the sample contract."""


def read_sample(path):
    """Return ``(arrays, attrs)`` for one sample, checking dtypes and shapes.

    Only the datasets and attributes named in the contract are read.
    """
    import h5py

    try:
        with h5py.File(path, 'r') as f:
            version = f.attrs.get('schema_version')
            if isinstance(version, bytes):
                version = version.decode()
            if version != SCHEMA_VERSION:
                raise SampleFormatError(
                    'schema_version {!r}, expected {!r}'.format(version, SCHEMA_VERSION))
            arrays = {}
            for name, (dtype, rank) in SAMPLE_DATASETS.items():
                if name not in f:
                    raise SampleFormatError('missing dataset {}'.format(name))
                arr = f[name][()]
                if arr.dtype != dtype or arr.ndim != rank:
                    raise SampleFormatError('{} is {} with rank {}, expected {} with rank {}'.format(
                        name, arr.dtype, arr.ndim, np.dtype(dtype), rank))
                arrays[name] = arr
            attrs = {
                'frequency_hz': float(f.attrs['frequency_hz']),
                'spacing_mm': float(f.attrs['spacing_mm']),
            }
    except (OSError, KeyError) as e:
        raise SampleFormatError(
            'unreadable sample: {}'.format(e.__class__.__name__))

    grid = arrays['ct_hu'].shape
    for name, arr in arrays.items():
        expected = (2,) + grid if name in COMPLEX_FIELDS else grid
        if arr.shape != expected:
            raise SampleFormatError('{} has shape {}, expected {}'.format(
                name, arr.shape, expected))
    return arrays, attrs


def torch_dataset(records):
    """A ``torch.utils.data.Dataset`` over store records.

    Each item is a dict of tensors named as in ``SAMPLE_DATASETS``, plus
    ``frequency_hz`` and ``spacing_mm``. Assembling model inputs is left to
    the model package, see SF-04.
    """
    import torch
    from torch.utils.data import Dataset

    class BabelBrainSamples(Dataset):

        def __init__(self, records):
            self.records = list(records)

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            arrays, attrs = read_sample(self.records[index].path)
            item = {name: torch.from_numpy(arr)
                    for name, arr in arrays.items()}
            item.update(attrs)
            return item

    return BabelBrainSamples(records)
