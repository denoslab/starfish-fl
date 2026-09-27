import hashlib
import os
import re
import tempfile
import zipfile

base_folder = '/starfish/artifacts'

zip_prefix = '.zip'

FILE_TYPES = ('artifacts', 'mid_artifacts', 'logs')
CHUNK_BYTES = 1 << 20
_SAFE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$')
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


class TransferError(Exception):
    """A rejected file transfer; ``status`` is the HTTP status to answer with."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def generate_url(run_id, task_seq, round_seq):
    if not run_id or not task_seq or not round_seq:
        return None
    return f"{base_folder}/{run_id}/{task_seq}/{round_seq}/"


def gen_batch_url(project_id, batch, task_seq, round_seq):
    """Folder for files shared by every run of a batch, such as an aggregated artifact."""
    if not project_id or not batch or not task_seq or not round_seq:
        return None
    return f"{base_folder}/batch-{project_id}-{batch}/{task_seq}/{round_seq}/"


def zip_all_files(run, url_list, file_type):
    """Zip the files into a temporary file on disk and return it opened for reading.

    The file is unlinked at once, so it disappears when the response closes it.
    Nothing is held in memory, whatever the size of the files.
    """
    if not url_list:
        return None
    folder = os.path.join(base_folder, 'tmp')
    os.makedirs(folder, exist_ok=True)
    fd, zip_path = tempfile.mkstemp(dir=folder, prefix='{}-{}-'.format(run.id, file_type),
                                    suffix=zip_prefix)
    try:
        with os.fdopen(fd, 'wb') as raw, zipfile.ZipFile(raw, 'w') as zip_file:
            for file_path in dict.fromkeys(url_list):
                zip_file.write(file_path, os.path.basename(file_path))
        handle = open(zip_path, 'rb')
    finally:
        os.unlink(zip_path)
    return handle


def path_seq_round(path):
    """Task seq and round of a stored file, from its folder: .../<seq>/<round>/<name>."""
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) < 3:
        return None, None
    return parts[-3], parts[-2]


def get_file_urls(runs, task_seq, round_seq, file_type) -> []:
    urls = []
    if not runs or len(runs) == 0 or not file_type:
        return urls
    for run in runs:
        if not run:
            continue
        saved_files = []
        if file_type == 'artifacts':
            saved_files = run.artifacts
        if file_type == 'logs':
            saved_files = run.logs
        if file_type == 'mid_artifacts':
            saved_files = run.middle_artifacts
            # if task_seq and round_seq not provided, which means users want to download all files under the run
        if task_seq and round_seq:
            picked_files = [item for item in saved_files
                            if path_seq_round(item) == (str(task_seq), str(round_seq))]
            urls.extend(picked_files)
        else:
            urls.extend(saved_files)
    # An aggregated artifact is shared by every run of a batch: list it once
    return list(dict.fromkeys(urls))


def gen_unique_file_name(file_name, run, cur_seq, cur_round):
    return '{}-{}-{}-{}'.format(run, cur_seq, cur_round, file_name)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(CHUNK_BYTES), b''):
            digest.update(block)
    return digest.hexdigest()


def check_upload_params(file_type, name, size, sha256, max_bytes):
    """Validate the query parameters of a streamed upload; return size as int."""
    if file_type not in FILE_TYPES:
        raise TransferError(
            'type must be one of {}'.format(', '.join(FILE_TYPES)))
    if not name or not _SAFE_NAME.match(name):
        raise TransferError('name must be a plain file name')
    if not sha256 or not _SHA256.match(sha256):
        raise TransferError('sha256 must be 64 lowercase hex characters')
    try:
        size = int(size)
    except (TypeError, ValueError):
        raise TransferError('size must be an integer')
    if size < 0:
        raise TransferError('size must not be negative')
    if size > max_bytes:
        raise TransferError('file of {} bytes exceeds the limit of {} bytes'.format(
            size, max_bytes), status=413)
    return size


def check_free_disk(size, reserve_bytes, disk_usage):
    """Refuse an upload that would leave less than ``reserve_bytes`` free."""
    os.makedirs(base_folder, exist_ok=True)
    free = disk_usage(base_folder).free
    if free - size < reserve_bytes:
        raise TransferError(
            'not enough free disk for {} bytes'.format(size), status=507)


def receive_stream(read, dest_dir, final_name, size, sha256):
    """Write a request body to ``dest_dir/final_name`` only if its size and SHA-256 match.

    ``read(n)`` returns up to n bytes of the body. The body goes to a temp file
    in chunks while it is hashed; on any mismatch the temp file is removed and
    nothing appears at the final path. Returns the final path.
    """
    os.makedirs(dest_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dest_dir, prefix='.upload-')
    digest = hashlib.sha256()
    received = 0
    try:
        with os.fdopen(fd, 'wb') as out:
            while received < size:
                block = read(min(CHUNK_BYTES, size - received))
                if not block:
                    break
                digest.update(block)
                out.write(block)
                received += len(block)
            if received == size and read(1):
                received += 1
        if received != size:
            raise TransferError(
                'received {} bytes, expected {}'.format(received, size))
        if digest.hexdigest() != sha256:
            raise TransferError('SHA-256 mismatch')
        final_path = os.path.join(dest_dir, final_name)
        os.replace(tmp_path, final_path)
        return final_path
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
