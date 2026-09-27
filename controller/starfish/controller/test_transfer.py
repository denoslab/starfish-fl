"""Tests for streamed artifact transfer, SF-02 part A.

A small in-process HTTP server stands in for the router's runs-action file
endpoints, so uploads and downloads go through real sockets.
"""

import hashlib
import json
import os
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import TestCase
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from starfish.controller.file import transfer
from starfish.controller.file.transfer import TransferFailed


class FakeRouter:
    """Stores uploaded files by name and serves them back, with faults on demand."""

    def __init__(self):
        self.files = {}
        self.fail_next = 0        # answer 503 to this many requests
        self.corrupt_next = 0     # flip bytes in this many downloads
        self.requests = []
        router = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _params(self):
                return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

            def _answer(self, code, body=b'', ctype='application/json'):
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _faulted(self):
                router.requests.append(
                    (self.command, urlparse(self.path).path))
                if router.fail_next:
                    router.fail_next -= 1
                    self._answer(503, b'busy')
                    return True
                return False

            def do_PUT(self):
                length = int(self.headers['Content-Length'])
                digest, received, chunks = hashlib.sha256(), 0, []
                while received < length:
                    block = self.rfile.read(min(65536, length - received))
                    digest.update(block)
                    chunks.append(block)
                    received += len(block)
                if self._faulted():
                    return
                p = self._params()
                if int(p['size']) != received or p['sha256'] != digest.hexdigest():
                    self._answer(400, b'"SHA-256 mismatch"')
                    return
                router.files[p['name']] = b''.join(chunks)
                self._answer(201, json.dumps({'name': p['name'], 'size': received,
                                              'sha256': p['sha256']}).encode())

            def do_GET(self):
                if self._faulted():
                    return
                path = urlparse(self.path).path
                if path.endswith('/files/'):
                    listing = [{'name': n, 'size': len(d), 'sha256': hashlib.sha256(d).hexdigest()}
                               for n, d in sorted(router.files.items())]
                    self._answer(200, json.dumps(listing).encode())
                    return
                data = router.files.get(self._params()['name'])
                if data is None:
                    self._answer(404, b'"not found"')
                    return
                if router.corrupt_next:
                    router.corrupt_next -= 1
                    data = bytes([data[0] ^ 0xFF]) + data[1:]
                self._answer(200, data, 'application/octet-stream')

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = 'http://127.0.0.1:{}'.format(self.server.server_address[1])
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class TransferTest(TestCase):

    def setUp(self):
        self.router = FakeRouter()
        self.addCleanup(self.router.close)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        env = patch.dict(os.environ, {'ROUTER_URL': self.router.url, 'ROUTER_USERNAME': 'u',
                                      'ROUTER_PASSWORD': 'p', 'STARFISH_TRANSFER_RETRIES': '3'})
        env.start()
        self.addCleanup(env.stop)
        sleep = patch.object(transfer.time, 'sleep')
        sleep.start()
        self.addCleanup(sleep.stop)

    def write(self, name, data):
        path = os.path.join(self.tmp, name)
        with open(path, 'wb') as f:
            f.write(data)
        return path

    def test_round_trip(self):
        data = os.urandom(3 * transfer.CHUNK_BYTES + 5)
        record = transfer.upload_file(self.write('mid', data), 7, 1, 1, 'mid_artifacts',
                                      name='mid-artifacts')
        self.assertEqual(record['sha256'], hashlib.sha256(data).hexdigest())
        out = os.path.join(self.tmp, 'out')
        paths = transfer.download_all(7, 'mid_artifacts', out)
        self.assertEqual(len(paths), 1)
        with open(paths[0], 'rb') as f:
            self.assertEqual(f.read(), data)
        self.assertEqual(os.listdir(out), ['mid-artifacts'])

    def test_upload_body_is_streamed_from_the_file(self):
        path = self.write('mid', b'abc' * 1000)
        with patch.object(transfer.requests, 'put', wraps=transfer.requests.put) as put:
            transfer.upload_file(path, 7, 1, 1, 'mid_artifacts')
        body = put.call_args.kwargs['data']
        self.assertTrue(hasattr(body, 'read'),
                        'upload must pass a file object, not bytes')

    def test_upload_retries_on_server_errors(self):
        self.router.fail_next = 2
        transfer.upload_file(self.write('mid', b'data'),
                             7, 1, 1, 'mid_artifacts')
        self.assertIn('mid', self.router.files)
        self.assertEqual(len(self.router.requests), 3)

    def test_upload_gives_up_after_the_retries(self):
        self.router.fail_next = 10
        with self.assertRaisesRegex(TransferFailed, '3 attempts'):
            transfer.upload_file(self.write('mid', b'data'),
                                 7, 1, 1, 'mid_artifacts')

    def test_refusal_is_not_retried(self):
        with patch.object(transfer, 'sha256_file', return_value='0' * 64):
            with self.assertRaisesRegex(TransferFailed, 'refused with 400'):
                transfer.upload_file(self.write(
                    'mid', b'data'), 7, 1, 1, 'mid_artifacts')
        self.assertEqual(len(self.router.requests), 1)

    def test_corrupted_download_is_retried_then_kept(self):
        self.router.files['artifacts'] = b'weights' * 100
        self.router.corrupt_next = 1
        entry = transfer.list_files(7, 'artifacts')[0]
        path = transfer.download_file(7, 'artifacts', entry, self.tmp)
        with open(path, 'rb') as f:
            self.assertEqual(f.read(), b'weights' * 100)

    def test_persistently_corrupted_download_leaves_nothing(self):
        self.router.files['artifacts'] = b'weights'
        self.router.corrupt_next = 10
        entry = transfer.list_files(7, 'artifacts')[0]
        out = os.path.join(self.tmp, 'out')
        with self.assertRaises(TransferFailed):
            transfer.download_file(7, 'artifacts', entry, out)
        self.assertEqual(os.listdir(out), [])

    def test_unsafe_names_in_a_listing_are_refused(self):
        for name in ('../escape', '.hidden', 'a/b'):
            with self.assertRaises(TransferFailed):
                transfer.download_file(7, 'artifacts', {'name': name, 'sha256': '0' * 64},
                                       self.tmp)

    def test_missing_router_url_fails_clearly(self):
        with patch.dict(os.environ, {'ROUTER_URL': ''}):
            with self.assertRaisesRegex(TransferFailed, 'ROUTER_URL'):
                transfer.list_files(7, 'logs')
