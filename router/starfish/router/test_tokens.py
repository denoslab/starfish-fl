"""Tests for enrolment codes, site tokens and per-site access, SF-09."""

import uuid
from datetime import timedelta

from django.utils import timezone
from rest_framework.test import APIClient

from starfish.router.auth import hash_secret
from starfish.router.models import EnrolmentCode, Run, Site, SiteToken
from starfish.router.test_transfer import BASE, TransferTestCase, fresh
from starfish import settings as router_settings

API = '/starfish/api/v1/'


class TokenTestCase(TransferTestCase):

    def setUp(self):
        super().setUp()
        self.tokens = {}
        for key, run in (('co', self.co), ('pa1', self.pa1), ('pa2', self.pa2)):
            site = Site.objects.get(uid=run.site_uid)
            secret = 'secret-{}'.format(key)
            SiteToken.objects.create(site=site, token_hash=hash_secret(secret))
            self.tokens[key] = secret
        # Each site uploads its delta as the superuser, before switching to tokens
        for run in self.runs:
            self.assertEqual(self.put_file(run, b'delta-%d' %
                             run.id).status_code, 201)

    def as_site(self, key):
        self.client.force_authenticate(None)
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.tokens[key])

    def files(self, run, all_runs='0'):
        return self.client.get(BASE + 'files/', {'run': run.id, 'type': 'mid_artifacts',
                                                 'all_runs': all_runs})


class EnrolmentTest(TokenTestCase):

    def code(self, **extra):
        self.client.force_authenticate(self.user)
        response = self.client.post(
            API + 'enrolment-codes/', extra, format='json')
        self.assertEqual(response.status_code, 201, response.data)
        return response.data['code']

    def enrol(self, code, uid=None, name='new-site'):
        client = APIClient()
        return client.post(API + 'sites/enrol/', {'code': code, 'uid': uid or str(uuid.uuid4()),
                                                  'name': name, 'description': 'd'}, format='json')

    def test_a_code_works_once(self):
        code = self.code(project=self.project.id)
        first = self.enrol(code)
        self.assertEqual(first.status_code, 201, first.data)
        self.assertTrue(first.data['token'])
        self.assertEqual(first.data['project'], self.project.id)
        self.assertEqual(self.enrol(code, name='second-site').status_code, 403)
        # The new token works
        client = APIClient()
        client.credentials(HTTP_AUTHORIZATION='Token ' + first.data['token'])
        self.assertEqual(client.get(API + 'projects/').status_code, 200)

    def test_expired_or_unknown_code_is_refused(self):
        code = self.code(valid_hours=1)
        EnrolmentCode.objects.update(
            expires_at=timezone.now() - timedelta(minutes=1))
        self.assertEqual(self.enrol(code).status_code, 403)
        self.assertEqual(self.enrol('not-a-code').status_code, 403)

    def test_codes_and_tokens_are_stored_hashed(self):
        code = self.code()
        token = self.enrol(code).data['token']
        self.assertFalse(EnrolmentCode.objects.filter(code_hash=code).exists())
        self.assertFalse(SiteToken.objects.filter(token_hash=token).exists())
        self.assertTrue(SiteToken.objects.filter(
            token_hash=hash_secret(token)).exists())

    def test_admin_endpoints_refuse_site_tokens(self):
        self.as_site('co')
        self.assertEqual(self.client.post(
            API + 'enrolment-codes/', {}, format='json').status_code, 403)
        self.assertEqual(self.client.get(
            API + 'site-tokens/').status_code, 403)


class RevocationTest(TokenTestCase):

    def test_revoked_token_is_rejected(self):
        self.as_site('pa1')
        self.assertEqual(self.files(self.pa1).status_code, 200)
        token = SiteToken.objects.get(
            token_hash=hash_secret(self.tokens['pa1']))
        self.client.credentials()
        self.client.force_authenticate(self.user)
        self.assertEqual(self.client.post(
            API + 'site-tokens/{}/revoke/'.format(token.id)).status_code, 200)
        self.as_site('pa1')
        self.assertEqual(self.files(self.pa1).status_code, 401)

    def test_unknown_token_is_rejected(self):
        self.client.force_authenticate(None)
        self.client.credentials(HTTP_AUTHORIZATION='Token nope')
        self.assertEqual(self.files(self.pa1).status_code, 401)


class AccessTest(TokenTestCase):

    def test_site_a_cannot_fetch_site_bs_mid_artifacts(self):
        self.as_site('pa1')
        self.assertEqual(self.files(self.pa2).status_code, 403)
        name = '{}-1-1-mid-artifacts'.format(self.pa2.id)
        for params in ({'run': self.pa2.id, 'type': 'mid_artifacts', 'name': name},
                       {'run': self.pa1.id, 'type': 'mid_artifacts', 'name': name, 'all_runs': '1'}):
            self.assertIn(self.client.get(BASE + 'file/',
                          params).status_code, (403, 404))
        self.assertEqual(self.client.get(BASE + 'download/', {
            'run': self.pa2.id, 'type': 'mid_artifacts'}).status_code, 403)

    def test_a_site_reads_its_own_files_and_the_coordinator_its_batch(self):
        self.as_site('pa1')
        self.assertEqual(len(self.files(self.pa1).data), 1)
        self.assertEqual(len(self.files(self.pa1, all_runs='1').data), 1)
        self.as_site('co')
        self.assertEqual(len(self.files(self.co, all_runs='1').data), 3)

    def test_a_site_cannot_upload_into_or_move_another_sites_run(self):
        self.as_site('pa1')
        self.assertEqual(self.put_file(self.pa2, b'x').status_code, 403)
        response = self.client.put(API + 'runs/{}/status/'.format(self.pa2.id),
                                   {'status': Run.RunStatus.PREPARING}, format='json')
        self.assertIn(response.status_code, (403, 404))
        self.assertEqual(fresh(self.pa2).status, Run.RunStatus.STANDBY)

    def test_a_site_sees_only_its_own_active_runs(self):
        self.as_site('pa1')
        ids = {r['id'] for r in self.client.get(API + 'runs/active/').data}
        self.assertEqual(ids, {self.pa1.id})

    def test_heartbeat_only_as_itself(self):
        self.as_site('pa1')
        own = str(Site.objects.get(uid=self.pa1.site_uid).uid)
        other = str(Site.objects.get(uid=self.pa2.site_uid).uid)
        self.assertEqual(self.client.post(API + 'sites/heartbeat/', {'uid': own, 'status': 1},
                                          format='json').status_code, 202)
        self.assertEqual(self.client.post(API + 'sites/heartbeat/', {'uid': other, 'status': 1},
                                          format='json').status_code, 403)

    def test_only_the_coordinator_site_starts_runs(self):
        self.as_site('pa1')
        self.assertEqual(self.client.post(API + 'runs', {'project': self.project.id},
                                          format='json').status_code, 403)

    def test_superuser_basic_auth_still_works(self):
        self.client.credentials()
        self.client.force_authenticate(self.user)
        self.assertEqual(len(self.files(self.co, all_runs='1').data), 3)


class TlsSettingsTest(TransferTestCase):

    def test_tls_settings(self):
        self.assertEqual(router_settings.tls_settings(False), {})
        on = router_settings.tls_settings(True)
        self.assertTrue(on['SECURE_SSL_REDIRECT'])
        self.assertTrue(on['SESSION_COOKIE_SECURE']
                        and on['CSRF_COOKIE_SECURE'])
        self.assertEqual(on['SECURE_PROXY_SSL_HEADER'],
                         ('HTTP_X_FORWARDED_PROTO', 'https'))
