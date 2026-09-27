"""Send a large random file to the router and back, and report memory and time.

Used by e2e/test_large_transfer.py inside a workbench controller::

    python -m starfish.controller.file.transfer_check --run <run id> --size-mb 2048

Prints one JSON line: sizes, hashes, seconds, and this process's peak RSS.
The local files are removed afterwards; the copy on the router stays.
"""

import argparse
import json
import os
import resource
import shutil
import sys
import tempfile
import time

from starfish.controller.file import transfer

NAME = 'transfer-check'


def write_random(path, size_mb):
    block = 1 << 20
    with open(path, 'wb') as f:
        for _ in range(size_mb):
            f.write(os.urandom(block))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--size-mb', type=int, default=2048)
    parser.add_argument('--folder', default=None)
    args = parser.parse_args(argv)

    folder = tempfile.mkdtemp(prefix='transfer-check-', dir=args.folder)
    try:
        source = os.path.join(folder, NAME)
        write_random(source, args.size_mb)
        start = time.time()
        record = transfer.upload_file(
            source, args.run, 1, 1, 'mid_artifacts', name=NAME)
        uploaded = time.time()
        entry = next(e for e in transfer.list_files(args.run, 'mid_artifacts', 1, 1)
                     if e['name'] == record['name'])
        copy = transfer.download_file(args.run, 'mid_artifacts', entry,
                                      os.path.join(folder, 'download'))
        downloaded = time.time()
        report = {
            'bytes': os.path.getsize(source),
            'router_name': record['name'],
            'match': transfer.sha256_file(copy) == transfer.sha256_file(source) == record['sha256'],
            'upload_s': round(uploaded - start, 1),
            'download_s': round(downloaded - uploaded, 1),
            # ru_maxrss is in KiB on Linux
            'peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        }
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    print(json.dumps(report))
    return 0 if report['match'] else 1


if __name__ == '__main__':
    sys.exit(main())
