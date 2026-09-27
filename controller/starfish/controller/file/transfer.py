"""Streamed artifact transfer between a controller and the router, SF-02 part A.

Files never pass through memory whole. ``upload_file`` hashes the file, then
sends it as a streamed request body; the router keeps it only if size and
SHA-256 match. ``download_file`` streams to a temp file next to the target
and moves it into place only after the SHA-256 from the router's listing
matches. Both retry the whole transfer a few times on network errors.

Typical use on a coordinator::

    for entry in list_files(run_id, 'mid_artifacts', task_seq, round_seq, all_runs=True):
        download_file(run_id, 'mid_artifacts', entry, folder, all_runs=True)
"""

import hashlib
import logging
import os
import tempfile
import time

import requests

CHUNK_BYTES = 1 << 20
DEFAULT_RETRIES = 3
TIMEOUT = (10, 300)

logger = logging.getLogger(__name__)


class TransferFailed(Exception):
    """A transfer that did not succeed after all retries, or was refused."""


def _router():
    url = os.getenv('ROUTER_URL')
    if not url:
        raise TransferFailed('ROUTER_URL is not set')
    return url.rstrip('/'), (os.getenv('ROUTER_USERNAME'), os.getenv('ROUTER_PASSWORD'))


def _retries():
    return int(os.getenv('STARFISH_TRANSFER_RETRIES', DEFAULT_RETRIES))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(CHUNK_BYTES), b''):
            digest.update(block)
    return digest.hexdigest()


class _Retryable(Exception):
    """A failure worth another attempt: a 5xx answer or a corrupted download."""


def _with_retries(what, attempt, retries, sleep=None):
    """Run ``attempt()``; retry on network errors and 5xx answers, not on 4xx."""
    last = None
    for n in range(1, retries + 1):
        try:
            return attempt()
        except (requests.RequestException, _Retryable) as e:
            last = e
        logger.warning('{} failed, attempt {} of {}: {}'.format(
            what, n, retries, last))
        if n < retries:
            (sleep or time.sleep)(min(2 ** n, 30))
    raise TransferFailed(
        '{} failed after {} attempts: {}'.format(what, retries, last))


def _check(response, what):
    if response.status_code >= 500:
        raise _Retryable('{} answered {}'.format(what, response.status_code))
    if response.status_code >= 400:
        raise TransferFailed('{} refused with {}: {}'.format(
            what, response.status_code, response.text[:200]))


def upload_file(path, run_id, task_seq, round_seq, file_type, name=None,
                retries=None, sleep=None):
    """Stream one file to the router. Returns the router's record: name, size, sha256."""
    router, auth = _router()
    size = os.path.getsize(path)
    sha256 = sha256_file(path)
    params = {'run': run_id, 'task_seq': task_seq, 'round_seq': round_seq, 'type': file_type,
              'name': name or os.path.basename(path), 'size': size, 'sha256': sha256}

    def attempt():
        with open(path, 'rb') as body:
            response = requests.put('{}/runs-action/file/'.format(router), params=params,
                                    data=body, auth=auth, timeout=TIMEOUT,
                                    headers={'Content-Type': 'application/octet-stream',
                                             'Content-Length': str(size)})
        _check(response, 'upload')
        record = response.json()
        if record.get('sha256') != sha256:
            raise TransferFailed('router recorded a different SHA-256')
        return record

    return _with_retries('upload of {}'.format(params['name']), attempt,
                         retries or _retries(), sleep)


def list_files(run_id, file_type, task_seq=None, round_seq=None, all_runs=False):
    """The router's listing: a list of dicts with name, size and sha256."""
    router, auth = _router()
    params = {'run': run_id, 'type': file_type,
              'all_runs': '1' if all_runs else '0'}
    if task_seq is not None and round_seq is not None:
        params.update(task_seq=task_seq, round_seq=round_seq)

    def attempt():
        response = requests.get('{}/runs-action/files/'.format(router), params=params,
                                auth=auth, timeout=TIMEOUT)
        _check(response, 'listing')
        return response.json()

    return _with_retries('listing of {}'.format(file_type), attempt, _retries())


def download_file(run_id, file_type, entry, folder, all_runs=False, retries=None,
                  sleep=None):
    """Stream one listed file into ``folder``; returns its path once its hash matches."""
    router, auth = _router()
    name, expected = entry['name'], entry['sha256']
    if os.path.basename(name) != name or name.startswith('.'):
        raise TransferFailed('refusing file name {!r}'.format(name))
    os.makedirs(folder, exist_ok=True)
    final_path = os.path.join(folder, name)
    params = {'run': run_id, 'type': file_type, 'name': name,
              'all_runs': '1' if all_runs else '0'}

    def attempt():
        fd, tmp_path = tempfile.mkstemp(dir=folder, prefix='.download-')
        digest = hashlib.sha256()
        try:
            with os.fdopen(fd, 'wb') as out, requests.get(
                    '{}/runs-action/file/'.format(router), params=params, auth=auth,
                    stream=True, timeout=TIMEOUT) as response:
                _check(response, 'download')
                for block in response.iter_content(CHUNK_BYTES):
                    digest.update(block)
                    out.write(block)
            if digest.hexdigest() != expected:
                raise _Retryable('SHA-256 mismatch for {}'.format(name))
            os.replace(tmp_path, final_path)
            return final_path
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return _with_retries('download of {}'.format(name), attempt, retries or _retries(), sleep)


def download_all(run_id, file_type, folder, task_seq=None, round_seq=None, all_runs=False):
    """Download every listed file; returns their paths."""
    return [download_file(run_id, file_type, entry, folder, all_runs=all_runs)
            for entry in list_files(run_id, file_type, task_seq, round_seq, all_runs)]


# ── model registry, SF-12 ─────────────────────────────────────────────────────

def publish_model(run_id, task_seq, round_seq, bucket_hz, source_version, parent=None,
                  eval_report=None):
    """Coordinator: register a round's aggregated artifact as an approved model version."""
    router, auth = _router()
    body = {'run': run_id, 'task_seq': task_seq, 'round_seq': round_seq, 'bucket_hz': bucket_hz,
            'source_version': source_version, 'parent': parent, 'eval_report': eval_report or {}}

    def attempt():
        response = requests.post('{}/registry/'.format(router), json=body, auth=auth,
                                 timeout=TIMEOUT)
        _check(response, 'publish')
        return response.json()

    return _with_retries('publish of {}'.format(source_version), attempt, _retries())


def list_models(bucket_hz):
    """Approved model versions for a bucket, newest first."""
    router, auth = _router()

    def attempt():
        response = requests.get('{}/registry/'.format(router), params={'bucket_hz': bucket_hz},
                                auth=auth, timeout=TIMEOUT)
        _check(response, 'registry listing')
        return response.json()

    return _with_retries('registry listing', attempt, _retries())


def latest_model(bucket_hz):
    """The newest approved model version for a bucket, or None."""
    router, auth = _router()

    def attempt():
        response = requests.get('{}/registry/latest/'.format(router),
                                params={'bucket_hz': bucket_hz}, auth=auth, timeout=TIMEOUT)
        if response.status_code == 404:
            return None
        _check(response, 'registry lookup')
        return response.json()

    return _with_retries('registry lookup', attempt, _retries())


def download_model(entry, folder, retries=None, sleep=None):
    """Stream an approved model into ``folder`` and keep it only if its SHA-256 matches."""
    router, auth = _router()
    version, expected = entry['version'], entry['sha256']
    if os.path.basename(version) != version or version.startswith('.'):
        raise TransferFailed('refusing model version {!r}'.format(version))
    os.makedirs(folder, exist_ok=True)
    final_path = os.path.join(folder, version + '.safetensors')

    def attempt():
        fd, tmp_path = tempfile.mkstemp(dir=folder, prefix='.download-')
        digest = hashlib.sha256()
        try:
            with os.fdopen(fd, 'wb') as out, requests.get(
                    '{}/registry/file/'.format(router), params={'version': version}, auth=auth,
                    stream=True, timeout=TIMEOUT) as response:
                _check(response, 'model download')
                for block in response.iter_content(CHUNK_BYTES):
                    digest.update(block)
                    out.write(block)
            if digest.hexdigest() != expected:
                raise _Retryable('SHA-256 mismatch for {}'.format(version))
            os.replace(tmp_path, final_path)
            return final_path
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return _with_retries('download of {}'.format(version), attempt, retries or _retries(), sleep)
