"""
Toy Step 2 runs and a toy crop for the BabelBrain FL demo. Nothing here is real.

``write_run`` writes a pair of tiny files named like BabelBrain's Step 2
outputs, ``<ID>_<Tx>_<kHz>kHz_<PPW>PPW_DataForSim.h5`` and its ``Water_`` twin.
They hold no simulation, only FocalLength, Aperture and a few ``demo_*``
numbers that describe a made-up skull.

``demo_crop`` stands in for Tayeb's crop and resampling code, which BabelBrain
does not have yet. It builds a sample in the contract layout from those
numbers: a skull slab of the given thickness and tilt in front of a Gaussian
focus, and a transcranial field that is the water field attenuated, delayed
and shifted by the slab. The mapping is simple enough for the stand-in model
to learn something in three rounds, so the demo shows the loss and the gate
moving. It says nothing about real skulls or about tFUS-FNO.
"""

import os

import numpy as np

SHAPE = (16, 16, 32)
SPACING_MM = 0.49
# A real 250 kHz field in water has a 6 mm wavelength, about 12 voxels here.
# The toy uses 48 voxels, so the tiny stand-in learns something in a few rounds.
WAVE_NUMBER = 2 * np.pi / 48
FRONT = 3  # first skull voxel along the beam


def write_run(folder, prefix, rng, skull_mm, tilt):
    """Write one toy run; return the DataForSim.h5 path."""
    import h5py
    os.makedirs(folder, exist_ok=True)
    full = os.path.join(folder, prefix + 'DataForSim.h5')
    water = os.path.join(folder, prefix + 'Water_DataForSim.h5')
    with h5py.File(full, 'w') as f:
        f['FocalLength'] = 0.05
        f['Aperture'] = 0.05
        f.attrs['demo_skull_mm'] = float(skull_mm)
        f.attrs['demo_tilt'] = float(tilt)
        f.attrs['demo_focus'] = np.array([rng.uniform(6, 10), rng.uniform(6, 10),
                                          rng.uniform(17, 24)])
        f.attrs['demo_seed'] = int(rng.integers(2 ** 31))
    with h5py.File(water, 'w') as f:
        f.attrs['demo'] = 'water'
    return full


def _focus(x, y, z, fx, fy, fz):
    r2 = ((x - fx) / 2.0) ** 2 + ((y - fy) / 2.0) ** 2 + ((z - fz) / 5.0) ** 2
    beam = np.exp(-((x - fx) ** 2 + (y - fy) ** 2) / 18.0)
    return 1e5 * np.exp(-r2 / 2) + 2e4 * beam


def demo_crop(full_sol_path, water_sol_path, bucket_hz):
    """A toy sample from the ``demo_*`` numbers of a toy run, named as in the contract."""
    import h5py
    with h5py.File(full_sol_path, 'r') as f:
        skull_mm = float(f.attrs['demo_skull_mm'])
        tilt = float(f.attrs['demo_tilt'])
        fx, fy, fz = (float(v) for v in f.attrs['demo_focus'])
        rng = np.random.default_rng(int(f.attrs['demo_seed']))
    x, y, z = np.indices(SHAPE).astype(np.float64)
    # Thickness in voxels, varying across the beam with the tilt
    thick = np.clip(skull_mm / SPACING_MM * (1 + tilt * (x - 7.5) / 8), 1, 10)
    skull = (z >= FRONT) & (z < FRONT + thick)
    ct = np.where(skull, 1400 + 250 * rng.standard_normal(SHAPE),
                  np.where(z < FRONT, 0.0, 40.0))
    phase = WAVE_NUMBER * z
    amp = _focus(x, y, z, fx, fy, fz)
    water = np.stack([amp * np.cos(phase), -amp * np.sin(phase)])
    # Through the skull: weaker, earlier, and pushed sideways by the tilt
    amp_skull = np.exp(-0.12 * thick) * _focus(x, y, z,
                                               fx + 2.5 * tilt, fy, fz - 0.3 * thick)
    phase_skull = phase - 0.3 * thick
    transcranial = np.stack(
        [amp_skull * np.cos(phase_skull), -amp_skull * np.sin(phase_skull)])
    return {
        'ct_hu': ct.astype(np.float32),
        'water_field': water.astype(np.float32),
        'skull_field': transcranial.astype(np.float32),
        'sos': np.where(skull, 2800.0, 1500.0).astype(np.float32),
        'attenuation': np.where(skull, 80.0, 0.5).astype(np.float32),
        'brain_mask': (z >= FRONT + thick + 1).astype(np.uint8),
    }
