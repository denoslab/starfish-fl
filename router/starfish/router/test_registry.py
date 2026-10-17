"""Tests for the model registry, SF-12."""

import hashlib
import os

from django.contrib.auth.models import User

from starfish.router.models import ModelVersion
from starfish.router.test_tokens import TokenTestCase
from starfish.router.test_transfer import TransferTestCase

REGISTRY = '/starfish/api/v1/registry/'


class RegistryTest(TransferTestCase):

    def publish_round(self, round_seq, data=b'weights', run=None, **extra):
        run = run or self.co
        self.assertEqual(self.put_file(self.co, data, file_type='artifacts', name='artifacts',
                                       round_seq=round_seq).status_code, 201)
        body = {'run': run.id, 'task_seq': 1, 'round_seq': round_seq, 'bucket_hz': 250000,
                'source_version': 'standin-250k-p{}-b1-t1-r{}'.format(self.project.id, round_seq)}
        body.update(extra)
        return self.client.post(REGISTRY, body, format='json')

    def review(self, version, decision='approve', note=''):
        return self.client.post(REGISTRY + decision + '/', {'version': version, 'note': note},
                                format='json')

    def test_publish_numbers_versions_per_bucket_with_parents(self):
        first = self.publish_round(1, b'round one')
        self.assertEqual(first.status_code, 201, first.data)
        self.assertEqual(first.data['version'], '250k-v0001')
        self.assertIsNone(first.data['parent'])
        second = self.publish_round(2, b'round two', parent='250k-v0001',
                                    eval_report={'accepted': True})
        self.assertEqual(second.data['version'], '250k-v0002')
        self.assertEqual(second.data['parent'], '250k-v0001')
        self.assertEqual(second.data['eval_report'], {'accepted': True})
        self.assertEqual(second.data['sha256'],
                         hashlib.sha256(b'round two').hexdigest())

    def test_only_the_coordinator_can_publish(self):
        response = self.publish_round(1, run=self.pa1)
        self.assertEqual(response.status_code, 403)
        self.assertEqual(ModelVersion.objects.count(), 0)

    def test_publishing_the_same_round_twice_returns_the_same_version(self):
        self.assertEqual(self.publish_round(1).status_code, 201)
        body = {'run': self.co.id, 'task_seq': 1, 'round_seq': 1, 'bucket_hz': 250000,
                'source_version': 'x'}
        again = self.client.post(REGISTRY, body, format='json')
        self.assertEqual(again.status_code, 200)
        self.assertEqual(again.data['version'], '250k-v0001')
        self.assertEqual(ModelVersion.objects.count(), 1)

    def test_round_without_artifact_or_unknown_parent_is_refused(self):
        body = {'run': self.co.id, 'task_seq': 1, 'round_seq': 7, 'bucket_hz': 250000,
                'source_version': 'x'}
        self.assertEqual(self.client.post(
            REGISTRY, body, format='json').status_code, 404)
        self.assertEqual(self.publish_round(
            1, parent='250k-v0099').status_code, 404)

    def test_list_latest_and_download_with_checksum(self):
        self.publish_round(1, b'one')
        self.publish_round(2, b'two', parent='250k-v0001')
        self.review('250k-v0001')
        self.review('250k-v0002')
        listing = self.client.get(REGISTRY, {'bucket_hz': 250000})
        self.assertEqual([v['version']
                         for v in listing.data], ['250k-v0002', '250k-v0001'])
        self.assertEqual(self.client.get(
            REGISTRY, {'bucket_hz': 500000}).data, [])
        latest = self.client.get(REGISTRY + 'latest/', {'bucket_hz': 250000})
        self.assertEqual(latest.data['version'], '250k-v0002')
        response = self.client.get(
            REGISTRY + 'file/', {'version': '250k-v0002'})
        body = b''.join(response.streaming_content)
        self.assertEqual(body, b'two')
        self.assertEqual(response['X-Starfish-SHA256'],
                         hashlib.sha256(b'two').hexdigest())
        self.assertEqual(self.client.get(REGISTRY + 'latest/',
                         {'bucket_hz': 750000}).status_code, 404)

    def test_registry_keeps_its_own_copy(self):
        self.publish_round(1, b'keep me')
        self.review('250k-v0001')
        for path in self.co.__class__.objects.get(pk=self.co.pk).artifacts:
            os.remove(path)
        response = self.client.get(
            REGISTRY + 'file/', {'version': '250k-v0001'})
        self.assertEqual(b''.join(response.streaming_content), b'keep me')

    def test_registry_requires_authentication(self):
        self.client.force_authenticate(None)
        self.assertIn(self.client.get(
            REGISTRY, {'bucket_hz': 250000}).status_code, (401, 403))

    def test_a_published_model_waits_for_sign_off(self):
        response = self.publish_round(1, b'one')
        self.assertEqual(response.data['status'], 'pending')
        self.assertIsNone(response.data['reviewed_at'])
        self.assertEqual(self.client.get(
            REGISTRY, {'bucket_hz': 250000}).data, [])
        self.assertEqual(self.client.get(
            REGISTRY + 'latest/', {'bucket_hz': 250000}).status_code, 404)
        pending = self.client.get(
            REGISTRY, {'bucket_hz': 250000, 'status': 'pending'})
        self.assertEqual([v['version']
                         for v in pending.data], ['250k-v0001'])

    def test_approval_releases_the_model_and_records_who_and_when(self):
        self.publish_round(1, b'one')
        response = self.review('250k-v0001', note='checked on the test set')
        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(response.data['status'], 'approved')
        self.assertEqual(response.data['reviewed_by'], 'admin')
        self.assertIsNotNone(response.data['reviewed_at'])
        self.assertEqual(
            response.data['review_note'], 'checked on the test set')
        latest = self.client.get(REGISTRY + 'latest/', {'bucket_hz': 250000})
        self.assertEqual(latest.data['version'], '250k-v0001')

    def test_latest_skips_pending_and_rejected_versions(self):
        self.publish_round(1, b'one')
        self.publish_round(2, b'two', parent='250k-v0001')
        self.publish_round(3, b'three', parent='250k-v0002')
        self.review('250k-v0001')
        self.review('250k-v0002', 'reject')
        latest = self.client.get(REGISTRY + 'latest/', {'bucket_hz': 250000})
        self.assertEqual(latest.data['version'], '250k-v0001')
        newest = self.client.get(
            REGISTRY + 'latest/', {'bucket_hz': 250000, 'status': 'any'})
        self.assertEqual(newest.data['version'], '250k-v0003')
        rejected = self.client.get(
            REGISTRY, {'bucket_hz': 250000, 'status': 'rejected'})
        self.assertEqual([v['version']
                         for v in rejected.data], ['250k-v0002'])

    def test_a_decision_is_final(self):
        self.publish_round(1)
        self.assertEqual(self.review(
            '250k-v0001', 'reject').status_code, 200)
        self.assertEqual(self.review('250k-v0001').status_code, 409)
        # The same decision again is harmless
        again = self.review('250k-v0001', 'reject')
        self.assertEqual(again.status_code, 200)
        self.assertEqual(again.data['status'], 'rejected')

    def test_review_of_unknown_version_or_bad_status_filter(self):
        self.assertEqual(self.review('250k-v0042').status_code, 404)
        self.assertEqual(self.client.get(
            REGISTRY, {'bucket_hz': 250000, 'status': 'maybe'}).status_code, 400)

    def test_a_staff_user_who_is_not_superuser_cannot_review(self):
        self.publish_round(1)
        self.client.force_authenticate(User.objects.create_user('lab', 'l@example.com', 'pw',
                                                                is_staff=True))
        self.assertEqual(self.review('250k-v0001').status_code, 403)
        self.assertEqual(self.client.get(
            REGISTRY, {'bucket_hz': 250000, 'status': 'pending'}).status_code, 403)

    def test_rows_from_before_sign_off_count_as_approved(self):
        # Migration 0013 marks existing rows approved; the model default is pending
        self.publish_round(1)
        ModelVersion.objects.update(status=ModelVersion.Status.APPROVED)
        self.assertEqual(self.client.get(REGISTRY + 'latest/',
                         {'bucket_hz': 250000}).data['version'], '250k-v0001')


class RegistrySignOffWithTokensTest(TokenTestCase):
    """With site tokens, only the project's coordinator site signs off, SF-09 and SF-12."""

    def setUp(self):
        super().setUp()
        self.client.force_authenticate(self.user)
        self.assertEqual(self.put_file(self.co, b'global', file_type='artifacts',
                                       name='artifacts').status_code, 201)
        body = {'run': self.co.id, 'task_seq': 1, 'round_seq': 1, 'bucket_hz': 250000,
                'source_version': 'x'}
        self.assertEqual(self.client.post(
            REGISTRY, body, format='json').status_code, 201)

    def review(self, decision='approve'):
        return self.client.post(REGISTRY + decision + '/', {'version': '250k-v0001'},
                                format='json')

    def test_a_participant_neither_sees_nor_approves_a_pending_model(self):
        self.as_site('pa1')
        self.assertEqual(self.client.get(
            REGISTRY, {'bucket_hz': 250000}).data, [])
        self.assertEqual(self.client.get(
            REGISTRY, {'bucket_hz': 250000, 'status': 'pending'}).status_code, 403)
        self.assertEqual(self.client.get(
            REGISTRY + 'file/', {'version': '250k-v0001'}).status_code, 404)
        self.assertEqual(self.review().status_code, 403)
        self.assertEqual(self.review('reject').status_code, 403)
        self.assertEqual(ModelVersion.objects.get().status, 'pending')

    def test_the_coordinator_site_tests_and_approves(self):
        self.as_site('co')
        pending = self.client.get(
            REGISTRY, {'bucket_hz': 250000, 'status': 'pending'})
        self.assertEqual([v['version']
                         for v in pending.data], ['250k-v0001'])
        download = self.client.get(
            REGISTRY + 'file/', {'version': '250k-v0001'})
        self.assertEqual(b''.join(download.streaming_content), b'global')
        response = self.review()
        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(response.data['reviewed_by'],
                         'site:{}'.format(self.co.site_uid))
        self.as_site('pa2')
        self.assertEqual(self.client.get(REGISTRY + 'latest/',
                         {'bucket_hz': 250000}).data['version'], '250k-v0001')
        download = self.client.get(
            REGISTRY + 'file/', {'version': '250k-v0001'})
        self.assertEqual(b''.join(download.streaming_content), b'global')
