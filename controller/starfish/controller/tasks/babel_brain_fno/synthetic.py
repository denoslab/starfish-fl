"""Synthetic BabelBrain sample stores for tests and the workbench.

Writes tiny, random samples that follow the sample contract v1 in
babelbrain-docs/specs/00-sample-contract.md, plus a matching manifest.
Nothing here comes from a real subject.

Usage from the command line::

    python -m starfish.controller.tasks.babel_brain_fno.synthetic <store_root> \\
        --groups 5 --per-group 4 --val-groups 1 --bucket 250000
"""

import argparse
import hashlib
import hmac
import json
import os
import uuid

import numpy as np

from starfish.controller.tasks.babel_brain_fno.store import (
    BUCKET_SPACING_MM,
    CROP_MM,
    MANIFEST_NAME,
    REGIONS,
    SCHEMA_VERSION,
    SCHEMA_VERSION_REGIONS,
    STORE_VERSION_DIR,
)

DEFAULT_SHAPE = (16, 16, 32)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_sample_file(path, sample_id, bucket_hz, shape=DEFAULT_SHAPE, rng=None,
                      schema_version=SCHEMA_VERSION):
    """Write one random sample in the contract's HDF5 layout."""
    import h5py

    rng = rng or np.random.default_rng()
    x, y, z = shape
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('ct_hu', data=rng.uniform(
            -1000, 2000, shape).astype(np.float32))
        f.create_dataset('water_field', data=rng.standard_normal(
            (2, x, y, z)).astype(np.float32))
        f.create_dataset('skull_field', data=rng.standard_normal(
            (2, x, y, z)).astype(np.float32))
        f.create_dataset('sos', data=rng.uniform(
            1400, 3000, shape).astype(np.float32))
        f.create_dataset('attenuation', data=rng.uniform(
            0, 100, shape).astype(np.float32))
        f.create_dataset('brain_mask', data=(
            rng.random(shape) > 0.5).astype(np.uint8))
        f.attrs['schema_version'] = schema_version
        f.attrs['sample_id'] = sample_id
        f.attrs['frequency_hz'] = float(bucket_hz)
        f.attrs['spacing_mm'] = BUCKET_SPACING_MM[bucket_hz]
        f.attrs['crop_mm'] = np.array(CROP_MM)


def append_manifest(store_root, entry):
    """Append one JSON line to the store manifest."""
    path = os.path.join(store_root, STORE_VERSION_DIR, MANIFEST_NAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def delete_sample(store_root, sample_id):
    """Delete a sample the way the contract says: remove the file, append a tombstone."""
    folder = os.path.join(store_root, STORE_VERSION_DIR)
    for bucket in BUCKET_SPACING_MM:
        path = os.path.join(folder, str(bucket), '{}.h5'.format(sample_id))
        if os.path.exists(path):
            os.remove(path)
    append_manifest(store_root, {'sample_id': sample_id, 'deleted': True})


def _salt(store_root):
    path = os.path.join(store_root, STORE_VERSION_DIR, '.salt')
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(os.urandom(32))
    with open(path, 'rb') as f:
        return f.read()


def write_synthetic_store(store_root, n_groups=5, per_group=4, val_groups=1,
                          bucket_hz=250000, shape=DEFAULT_SHAPE, seed=0, regions=False):
    """Write a store with ``n_groups * per_group`` samples and return the manifest entries.

    The first ``val_groups`` groups go to ``val``, the rest to ``train``. The
    real exporter assigns splits by hashing ``group_id``; a fixed assignment
    keeps test counts predictable. With ``regions``, the store is a schema 1.1
    test set: samples cycle through the region classes, as a NeuroFUS
    eval-store export would label them.
    """
    version = SCHEMA_VERSION_REGIONS if regions else SCHEMA_VERSION
    count = 0
    rng = np.random.default_rng(seed)
    salt = _salt(store_root)
    entries = []
    for g in range(n_groups):
        subject = 'synthetic-subject-{:03d}'.format(g).encode()
        group_id = hmac.new(salt, subject, hashlib.sha256).hexdigest()
        split = 'val' if g < val_groups else 'train'
        for _ in range(per_group):
            sample_id = str(uuid.UUID(bytes=rng.bytes(16), version=4))
            rel = '{}/{}.h5'.format(bucket_hz, sample_id)
            path = os.path.join(store_root, STORE_VERSION_DIR, rel)
            write_sample_file(path, sample_id, bucket_hz, shape, rng,
                              schema_version=version)
            entry = {
                'sample_id': sample_id,
                'schema_version': version,
                'file': rel,
                'sha256': _sha256(path),
                'frequency_hz': bucket_hz,
                'bucket_hz': bucket_hz,
                'ppw': 9,
                'spacing_mm': BUCKET_SPACING_MM[bucket_hz],
                'babelbrain_version': '0.8.1',
                'tx_system': 'Single',
                'focal_length_mm': 50.0,
                'aperture_mm': 50.0,
                'ct_type': 'CT',
                'group_id': group_id,
                'split': split,
                'exported_month': '2026-10',
                'source': 'live',
                'fingerprint': hashlib.sha256(rng.bytes(32)).hexdigest(),
            }
            if regions:
                entry['region'] = REGIONS[count % len(REGIONS)]
                entry['source'] = 'backfill'
            count += 1
            append_manifest(store_root, entry)
            entries.append(entry)
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Write a synthetic BabelBrain FL sample store.')
    parser.add_argument('store_root')
    parser.add_argument('--groups', type=int, default=5)
    parser.add_argument('--per-group', type=int, default=4)
    parser.add_argument('--val-groups', type=int, default=1)
    parser.add_argument('--bucket', type=int, default=250000,
                        choices=sorted(BUCKET_SPACING_MM))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--regions', action='store_true',
                        help='write a schema 1.1 test set with region labels')
    args = parser.parse_args(argv)
    entries = write_synthetic_store(
        args.store_root, args.groups, args.per_group, args.val_groups,
        args.bucket, seed=args.seed, regions=args.regions)
    print('Wrote {} synthetic samples'.format(len(entries)))


if __name__ == '__main__':
    main()
