"""Read a site's BabelBrain sample store in place.

The store is written by BabelBrain's exporter in the layout of the sample
contract v1, babelbrain-docs/specs/00-sample-contract.md. Its location comes
only from this site's ``BABELBRAIN_FL_STORE`` environment variable, never
from task config, so a coordinator cannot point a participant at other files.

This module needs no torch and no h5py: it parses the manifest, applies
tombstones, validates each line against the manifest schema, keeps every
file inside the store, and verifies SHA-256 hashes, caching good results.

Task logs are uploaded to the router, so nothing here logs paths. Warnings
name a sample by its random ``sample_id`` or by manifest line number only.
"""

import hashlib
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

STORE_ENV = 'BABELBRAIN_FL_STORE'
STORE_VERSION_DIR = 'v1'
MANIFEST_NAME = 'manifest.jsonl'
SCHEMA_VERSION = '1.0'
SCHEMA_PATH = Path(__file__).resolve().parent / 'manifest.schema.json'

BUCKET_SPACING_MM = {250000: 0.490, 500000: 0.368, 750000: 0.245}
BUCKETS_HZ = tuple(sorted(BUCKET_SPACING_MM))
SPLITS = ('train', 'val')
CROP_MM = (41.2, 41.2, 82.4)

module_logger = logging.getLogger(__name__)


class StoreError(Exception):
    """Raised when the store cannot be opened at all."""


@dataclass(frozen=True)
class SampleRecord:
    """One usable sample. ``path`` is local only; never log or upload it."""
    sample_id: str
    path: str = field(repr=False)
    bucket_hz: int
    split: str
    group_id: str
    sha256: str


def _load_validator():
    from jsonschema import Draft202012Validator, FormatChecker
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _sha256(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk_size), b''):
            digest.update(block)
    return digest.hexdigest()


class HashCache:
    """Remembers files that passed the SHA-256 check.

    Keyed by ``sample_id`` and valid only while the file's size and mtime
    are unchanged. Holds no paths. With ``path=None`` it lives in memory.
    """

    def __init__(self, path=None):
        self.path = path
        self._entries = {}
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._entries = data
            except (OSError, ValueError):
                self._entries = {}

    @staticmethod
    def _stamp(stat):
        return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}

    def is_verified(self, sample_id, sha256, stat):
        entry = self._entries.get(sample_id)
        return bool(entry) and entry.get('sha256') == sha256 and \
            {k: entry.get(k)
             for k in ('size', 'mtime_ns')} == self._stamp(stat)

    def mark_verified(self, sample_id, sha256, stat):
        self._entries[sample_id] = dict(self._stamp(stat), sha256=sha256)

    def forget(self, sample_id):
        self._entries.pop(sample_id, None)

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self._entries, f)
        os.replace(tmp, self.path)


class SampleStore:
    """A site's sample store. Call :meth:`samples` or :meth:`counts` to use it."""

    def __init__(self, root, cache_path=None, logger=None):
        self.root = os.path.realpath(root)
        self.version_root = os.path.realpath(
            os.path.join(self.root, STORE_VERSION_DIR))
        self.logger = logger or module_logger
        self.cache = HashCache(cache_path)
        self.rejections = Counter()
        self.deleted = 0
        self._records = None

    @classmethod
    def from_env(cls, cache_path=None, logger=None, environ=None):
        """Open the store named by ``BABELBRAIN_FL_STORE``."""
        environ = os.environ if environ is None else environ
        root = environ.get(STORE_ENV)
        if not root:
            raise StoreError('{} is not set on this site'.format(STORE_ENV))
        return cls(root, cache_path=cache_path, logger=logger)

    def _reject(self, reason, who):
        self.rejections[reason] += 1
        self.logger.warning('Excluded {}: {}'.format(who, reason))

    def _read_manifest(self):
        path = os.path.join(self.version_root, MANIFEST_NAME)
        if not os.path.isfile(path):
            raise StoreError('store has no {}/{}'.format(
                STORE_VERSION_DIR, MANIFEST_NAME))
        validator = _load_validator()
        try:
            with open(path) as f:
                lines = f.readlines()
        except OSError as e:
            raise StoreError('cannot read the manifest: {}'.format(e.strerror))
        live, deleted = {}, set()
        for line_no, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            who = 'manifest line {}'.format(line_no)
            try:
                entry = json.loads(line)
            except ValueError:
                self._reject('invalid_json', who)
                continue
            if not validator.is_valid(entry):
                self._reject('schema', who)
                continue
            sample_id = entry['sample_id']
            if entry.get('deleted'):
                deleted.add(sample_id)
            elif sample_id in live:
                self._reject('duplicate', 'sample {}'.format(sample_id))
            else:
                live[sample_id] = entry
        # A tombstone wins wherever it appears: deleted samples never come back.
        for sample_id in deleted:
            if live.pop(sample_id, None) is not None:
                self.deleted += 1
                self.cache.forget(sample_id)
        return live

    def _check(self, entry):
        """Return a SampleRecord, or None after recording why the sample is excluded."""
        sample_id = entry['sample_id']
        who = 'sample {}'.format(sample_id)
        bucket_dir = entry['file'].split('/', 1)[0]
        if bucket_dir != str(entry['bucket_hz']):
            self._reject('bucket_mismatch', who)
            return None
        path = os.path.realpath(os.path.join(self.version_root, entry['file']))
        if os.path.commonpath([path, self.version_root]) != self.version_root:
            self._reject('outside_store', who)
            return None
        try:
            stat = os.stat(path)
        except OSError:
            self._reject('missing_file', who)
            return None
        if not self.cache.is_verified(sample_id, entry['sha256'], stat):
            if _sha256(path) != entry['sha256']:
                self.cache.forget(sample_id)
                self._reject('sha256_mismatch', who)
                return None
            self.cache.mark_verified(sample_id, entry['sha256'], stat)
        return SampleRecord(
            sample_id=sample_id, path=path, bucket_hz=entry['bucket_hz'],
            split=entry['split'], group_id=entry['group_id'],
            sha256=entry['sha256'])

    def scan(self):
        """Read the manifest and check every file. Safe to call again after changes."""
        self.rejections = Counter()
        self.deleted = 0
        records = []
        for entry in self._read_manifest().values():
            record = self._check(entry)
            if record is not None:
                records.append(record)
        try:
            self.cache.save()
        except OSError as e:
            self.logger.warning(
                'Could not save the hash cache: {}'.format(e.strerror))
        self._records = records
        if self.rejections:
            self.logger.warning('Store rejections by reason: {}'.format(
                dict(sorted(self.rejections.items()))))
        return records

    def samples(self, bucket_hz=None, split=None):
        """Usable samples, optionally filtered by bucket and split."""
        if self._records is None:
            self.scan()
        return [r for r in self._records
                if (bucket_hz is None or r.bucket_hz == bucket_hz)
                and (split is None or r.split == split)]

    def counts(self, bucket_hz=None):
        """Number of usable samples per split."""
        found = Counter(r.split for r in self.samples(bucket_hz))
        return {s: found.get(s, 0) for s in SPLITS}
