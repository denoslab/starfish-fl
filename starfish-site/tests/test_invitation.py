"""Invitation files, the invite command and the errors of enrolment."""

import contextlib
import datetime
import io
import json
import os
import shutil
import stat
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from starfish_site import cli
from starfish_site import config as C
from starfish_site import invitation as I

PEM = '-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n'


def answer(status, body=None, text=''):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = body
    response.text = text
    return response


class InvitationFileTest(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def write(self, data):
        path = os.path.join(self.folder, 'x' + I.SUFFIX)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(data if isinstance(data, str) else json.dumps(data))
        return path

    def test_an_invitation_survives_a_round_trip_and_stays_private(self):
        expires = datetime.datetime(2026, 11, 29, 10, 0, tzinfo=datetime.timezone.utc)
        sent = I.Invitation(router='https://10.0.0.5:8443/starfish/api/v1', code='-abc', expires_at=expires,
                            project=4, testbed='Testbed', note='Ask Room 2.14', ca_cert=PEM)
        path = I.write(os.path.join(self.folder, 'Hospital A'), sent)
        self.assertEqual(path.name, 'Hospital A' + I.SUFFIX)
        self.assertEqual(I.read(path), sent)
        if os.name != 'nt':
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)

    def test_a_bare_host_gets_the_api_path_and_a_z_time_reads_as_utc(self):
        found = I.read(self.write({'kind': I.KIND, 'version': 1, 'router': 'http://127.0.0.1:18100',
                                   'code': ' c0de ', 'expires_at': '2026-11-29T10:00:00.5Z'}))
        self.assertEqual(found.router, 'http://127.0.0.1:18100/starfish/api/v1')
        self.assertEqual(found.code, 'c0de')
        self.assertEqual(found.expires_at.utcoffset(), datetime.timedelta(0))
        self.assertIsNone(found.ca_cert)

    def test_expiry(self):
        past = I.Invitation('https://h/x', 'c', datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc))
        self.assertTrue(past.expired())
        self.assertFalse(I.Invitation('https://h/x', 'c').expired())
        self.assertEqual(I.describe_time(None), 'no date')

    def test_files_that_are_no_invitation_are_refused_with_a_reason(self):
        good = {'kind': I.KIND, 'version': 1, 'router': 'https://h:1', 'code': 'c'}
        cases = (('not json', 'not an invitation'),
                 ({'kind': 'other'}, 'not an invitation'),
                 (dict(good, version=2), 'another version'),
                 (dict(good, code=''), 'lacks'),
                 (dict(good, router='h:1'), 'scheme'),
                 (dict(good, ca_cert='junk'), 'certificate'),
                 (dict(good, project='4'), 'not a number'),
                 (dict(good, expires_at='soon'), 'cannot be read'))
        for data, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaises(I.InvitationError) as refused:
                    I.read(self.write(data))
                self.assertIn(reason, str(refused.exception))

    def test_a_large_or_missing_file_is_refused(self):
        path = self.write(' ' * (I.MAX_BYTES + 1))
        with self.assertRaises(I.InvitationError):
            I.read(path)
        with self.assertRaises(I.InvitationError):
            I.read(os.path.join(self.folder, 'missing' + I.SUFFIX))


class InviteCommandTest(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.out = os.path.join(self.folder, 'Hospital A')
        self.ca = os.path.join(self.folder, 'ca.pem')
        with open(self.ca, 'w') as f:
            f.write(PEM)

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def invite(self, response, *extra):
        argv = ['invite', '--router', 'https://10.0.0.5:8443', '--user', 'admin', '--out', self.out,
                '--testbed', 'Testbed', '--note', 'Room 2.14', *extra]
        out = io.StringIO()
        post = patch('requests.post', side_effect=response) if isinstance(response, Exception) else \
            patch('requests.post', return_value=response)
        with post as sent, patch.dict(os.environ, {'ROUTER_PASSWORD': 'pw'}), contextlib.redirect_stdout(out):
            code = cli.main(argv)
        return code, sent, out.getvalue()

    def test_the_invitation_carries_the_code_the_note_and_the_certificate(self):
        code, sent, out = self.invite(answer(201, {'id': 3, 'code': 'C0DE', 'project': 4,
                                                   'expires_at': '2026-11-29T10:00:00Z'}),
                                      '--project', '4', '--ca-cert', self.ca, '--valid-hours', '24')
        self.assertEqual(code, 0)
        self.assertEqual(sent.call_args.args[0], 'https://10.0.0.5:8443/starfish/api/v1/enrolment-codes/')
        self.assertEqual(sent.call_args.kwargs['auth'], ('admin', 'pw'))
        self.assertEqual(sent.call_args.kwargs['json'], {'valid_hours': 24.0, 'note': 'Room 2.14', 'project': 4})
        self.assertEqual(sent.call_args.kwargs['verify'], self.ca)
        found = I.read(self.out + I.SUFFIX)
        self.assertEqual((found.code, found.project, found.note, found.testbed), ('C0DE', 4, 'Room 2.14', 'Testbed'))
        self.assertEqual(found.ca_cert, PEM)
        self.assertIn('works once', out)

    def test_refusals_say_why(self):
        for response, reason in ((answer(401), 'or its password'), (answer(403), 'not an admin'),
                                 (answer(404), 'no project'),
                                 (requests.ConnectionError('down'), 'cannot be reached')):
            with self.subTest(reason=reason):
                with self.assertRaises(SystemExit) as stop:
                    self.invite(response, '--project', '9')
                self.assertIn(reason, str(stop.exception))
        self.assertFalse(os.path.exists(self.out + I.SUFFIX))


class EnrolErrorTest(unittest.TestCase):

    def setUp(self):
        self.home = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.home, ignore_errors=True)

    def enrol(self, response, router='https://router.lan:8443', **kwargs):
        post = patch('requests.post', side_effect=response) if isinstance(response, Exception) else \
            patch('requests.post', return_value=response)
        with post:
            return cli.enrol_site(self.home, router, 'C0DE', 'Lab B', **kwargs)

    def test_each_failure_has_its_kind(self):
        for response, router, kind in ((answer(403), 'https://router.lan:8443', 'refused'),
                                       (answer(500, text='boom'), 'https://router.lan:8443', 'failed'),
                                       (requests.ConnectionError('x'), 'https://router.lan:8443', 'unreachable'),
                                       (requests.exceptions.SSLError('x'), 'https://router.lan:8443', 'certificate'),
                                       (answer(201), 'router.lan', 'url')):
            with self.subTest(kind=kind):
                with self.assertRaises(cli.EnrolError) as failed:
                    self.enrol(response, router)
                self.assertEqual(failed.exception.kind, kind)
        self.assertEqual(C.read_config(self.home), {})

    def test_an_enrolled_site_needs_force(self):
        body = {'site': 7, 'uid': 'u', 'token': 'tok', 'project': None}
        answer_, path = self.enrol(answer(201, body))
        self.assertEqual(answer_['token'], 'tok')
        self.assertTrue(os.path.isfile(path))
        with self.assertRaises(cli.EnrolError) as failed:
            self.enrol(answer(201, body))
        self.assertEqual(failed.exception.kind, 'enrolled')
        self.enrol(answer(201, dict(body, token='tok2')), force=True)
        self.assertEqual(C.read_config(self.home)['ROUTER_TOKEN'], 'tok2')


if __name__ == '__main__':
    unittest.main()
