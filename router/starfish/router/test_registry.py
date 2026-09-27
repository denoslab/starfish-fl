"""Tests for the model registry, SF-12."""

import hashlib
import os

from starfish.router.models import ModelVersion
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
        for path in self.co.__class__.objects.get(pk=self.co.pk).artifacts:
            os.remove(path)
        response = self.client.get(
            REGISTRY + 'file/', {'version': '250k-v0001'})
        self.assertEqual(b''.join(response.streaming_content), b'keep me')

    def test_registry_requires_authentication(self):
        self.client.force_authenticate(None)
        self.assertIn(self.client.get(
            REGISTRY, {'bucket_hz': 250000}).status_code, (401, 403))
