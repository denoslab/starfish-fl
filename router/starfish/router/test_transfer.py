"""Tests for streamed artifact transfer, SF-02 part A."""

import hashlib
import io
import os
import shutil
import tempfile
import zipfile
from collections import namedtuple
from unittest.mock import patch
from uuid import uuid4

from django.contrib.auth.models import User
from django.test import override_settings
from django.utils import timezone
from rest_framework.test import APITestCase

from starfish.router.models import Project, ProjectParticipant, Run, Site, StoredFile
from starfish.utils import file_util

BASE = '/starfish/api/v1/runs-action/'
DiskUsage = namedtuple('DiskUsage', 'total used free')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fresh(run):
    """Reload a run; refresh_from_db is refused by the protected FSM status field."""
    return Run.objects.get(pk=run.pk)


def make_runs(project, participants):
    """Create one batch of runs the way BulkCreateRunAPIView does."""
    project.batch += 1
    project.save()
    now = timezone.now()
    return Run.objects.bulk_create([Run(
        project=project, participant=pp, site_uid=pp.site.uid, role=pp.role,
        status=Run.RunStatus.STANDBY, tasks=project.tasks, batch=project.batch, cur_seq=1,
        created_at=now, updated_at=now) for pp in participants])


class TransferTestCase(APITestCase):
    """A project with a coordinator and two participants, one batch of runs."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        patcher = patch.object(file_util, 'base_folder', self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.tmp, True)

        self.user = User.objects.create_superuser(
            'admin', 'a@example.com', 'pw')
        self.client.force_authenticate(self.user)
        sites = [Site.objects.create(name='site-{}'.format(i), description='d', uid=uuid4(),
                                     owner=self.user) for i in range(3)]
        self.project = Project.objects.create(name='p', description='d', site=sites[0], batch=0,
                                              tasks=[{'seq': 1, 'model': 'X', 'config': {}}])
        roles = [ProjectParticipant.Role.COORDINATOR] + \
            [ProjectParticipant.Role.PARTICIPANT] * 2
        participants = [ProjectParticipant.objects.create(
            site=site, project=self.project, role=role, notes='') for site, role in zip(sites, roles)]
        self.runs = make_runs(self.project, participants)
        self.co, self.pa1, self.pa2 = self.runs

    def put_file(self, run, data, file_type='mid_artifacts', name='mid-artifacts', **overrides):
        params = {'run': run.id, 'task_seq': 1, 'round_seq': 1, 'type': file_type,
                  'name': name, 'size': len(data), 'sha256': sha(data)}
        params.update(overrides)
        query = '&'.join('{}={}'.format(k, v) for k, v in params.items())
        return self.client.put(BASE + 'file/?' + query, data=data,
                               content_type='application/octet-stream')

    def list_files(self, run, file_type, all_runs=0, **extra):
        params = {'run': run.id, 'type': file_type, 'all_runs': all_runs}
        params.update(extra)
        return self.client.get(BASE + 'files/', params)

    def fetch(self, run, file_type, name, all_runs=0):
        response = self.client.get(BASE + 'file/', {'run': run.id, 'type': file_type,
                                                    'name': name, 'all_runs': all_runs})
        body = b''.join(
            response.streaming_content) if response.status_code == 200 else None
        return response, body


class UploadTest(TransferTestCase):

    def test_upload_records_file_and_hash(self):
        data = os.urandom(3 * file_util.CHUNK_BYTES + 17)
        response = self.put_file(self.pa1, data)
        self.assertEqual(response.status_code, 201, response.data)
        pa1 = fresh(self.pa1)
        self.assertEqual(len(pa1.middle_artifacts), 1)
        path = pa1.middle_artifacts[0]
        with open(path, 'rb') as f:
            self.assertEqual(f.read(), data)
        stored = StoredFile.objects.get(path=path)
        self.assertEqual((stored.size, stored.sha256), (len(data), sha(data)))

    def test_wrong_hash_is_rejected_and_nothing_recorded(self):
        data = b'x' * 1000
        response = self.put_file(self.pa1, data, sha256=sha(b'other'))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(fresh(self.pa1).middle_artifacts, [])
        self.assertEqual(StoredFile.objects.count(), 0)
        leftovers = [f for _, _, files in os.walk(self.tmp) for f in files]
        self.assertEqual(leftovers, [])

    def test_wrong_size_is_rejected(self):
        response = self.put_file(self.pa1, b'x' * 100, size=99)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(fresh(self.pa1).middle_artifacts, [])

    def test_unsafe_names_and_types_are_rejected(self):
        for overrides in ({'name': '..%2F..%2Fetc%2Fpasswd'}, {'name': '.hidden'},
                          {'type': 'dataset'}, {'sha256': 'ABC'}, {'size': '-1'}):
            response = self.put_file(self.pa1, b'data', **overrides)
            self.assertEqual(response.status_code, 400, overrides)

    @override_settings(STARFISH_MAX_ARTIFACT_BYTES=10)
    def test_file_over_the_limit_is_rejected(self):
        self.assertEqual(self.put_file(self.pa1, b'x' * 11).status_code, 413)

    def test_upload_refused_when_disk_is_nearly_full(self):
        with patch('starfish.router.views.shutil.disk_usage',
                   return_value=DiskUsage(10, 10, 100)):
            self.assertEqual(self.put_file(
                self.pa1, b'x' * 10).status_code, 507)

    def test_unknown_run_is_rejected(self):
        self.co.id = 99999
        self.assertEqual(self.put_file(self.co, b'data').status_code, 404)

    def test_upload_requires_authentication(self):
        self.client.force_authenticate(None)
        self.assertIn(self.put_file(self.pa1, b'data').status_code, (401, 403))
        self.assertIn(self.list_files(
            self.pa1, 'logs').status_code, (401, 403))


class AggregatedArtifactTest(TransferTestCase):

    def test_streamed_artifact_is_stored_once_for_the_batch(self):
        data = os.urandom(4096)
        self.assertEqual(self.put_file(self.co, data, file_type='artifacts',
                                       name='artifacts').status_code, 201)
        runs = [fresh(r) for r in self.runs]
        paths = set()
        for run in runs:
            self.assertEqual(len(run.artifacts), 1)
            paths.add(run.artifacts[0])
        self.assertEqual(len(paths), 1)
        on_disk = [f for _, _, files in os.walk(self.tmp) for f in files]
        self.assertEqual(len(on_disk), 1)
        for run in runs:
            response, body = self.fetch(
                run, 'artifacts', os.path.basename(run.artifacts[0]))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(body, data)

    def test_old_upload_stores_artifact_once_and_only_in_this_project(self):
        other_site = Site.objects.create(
            name='other', description='d', uid=uuid4(), owner=self.user)
        other_project = Project.objects.create(
            name='q', description='d', site=other_site, batch=0)
        other_run, = make_runs(other_project, [ProjectParticipant.objects.create(
            site=other_site, project=other_project, role=ProjectParticipant.Role.COORDINATOR, notes='')])
        self.assertEqual(other_run.batch, self.co.batch)

        upload = io.BytesIO(b'weights')
        upload.name = 'artifacts'
        response = self.client.post(BASE + 'upload/', {
            'run': self.co.id, 'task_seq': 1, 'round_seq': 1, 'artifacts': upload}, format='multipart')
        self.assertEqual(response.status_code, 200)
        runs = [fresh(r) for r in self.runs]
        for run in runs:
            self.assertEqual(len(run.artifacts), 1)
        self.assertEqual(len({r.artifacts[0] for r in runs}), 1)
        self.assertEqual(fresh(other_run).artifacts, [])

        # Participants download it through the old zip endpoint too
        response = self.client.get(BASE + 'download/', {
            'run': self.pa2.id, 'type': 'artifacts', 'task_seq': 1, 'round_seq': 1})
        self.assertEqual(response.status_code, 200)
        archive = zipfile.ZipFile(io.BytesIO(
            b''.join(response.streaming_content)))
        self.assertEqual([archive.read(n)
                         for n in archive.namelist()], [b'weights'])


class ListAndDownloadTest(TransferTestCase):

    def setUp(self):
        super().setUp()
        self.data = {run.id: os.urandom(2048) for run in self.runs}
        for run in self.runs:
            self.assertEqual(self.put_file(
                run, self.data[run.id]).status_code, 201)

    def test_listing_gives_names_sizes_and_hashes(self):
        response = self.list_files(self.pa1, 'mid_artifacts')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, [{'name': '{}-1-1-mid-artifacts'.format(self.pa1.id),
                                          'size': 2048, 'sha256': sha(self.data[self.pa1.id])}])

    def test_coordinator_lists_every_run_participant_only_its_own(self):
        self.assertEqual(
            len(self.list_files(self.co, 'mid_artifacts', all_runs=1).data), 3)
        self.assertEqual(
            len(self.list_files(self.pa1, 'mid_artifacts', all_runs=1).data), 1)

    def test_listing_filters_by_task_and_round(self):
        self.assertEqual(self.list_files(self.co, 'mid_artifacts', all_runs=1,
                                         task_seq=1, round_seq=2).data, [])
        self.assertEqual(len(self.list_files(self.co, 'mid_artifacts', all_runs=1,
                                             task_seq=1, round_seq=1).data), 3)

    def test_download_streams_the_file_with_its_hash(self):
        name = '{}-1-1-mid-artifacts'.format(self.pa2.id)
        response, body = self.fetch(self.co, 'mid_artifacts', name, all_runs=1)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, self.data[self.pa2.id])
        self.assertEqual(response['X-Starfish-SHA256'],
                         sha(self.data[self.pa2.id]))

    def test_participant_cannot_fetch_another_runs_file(self):
        name = '{}-1-1-mid-artifacts'.format(self.pa2.id)
        response, _ = self.fetch(self.pa1, 'mid_artifacts', name, all_runs=1)
        self.assertEqual(response.status_code, 404)

    def test_old_download_of_all_runs_returns_every_file(self):
        """Before SF-02, omitting task_seq and round_seq returned only the last run's files."""
        response = self.client.get(BASE + 'download/', {
            'run': self.co.id, 'type': 'mid_artifacts', 'all_runs': '1'})
        self.assertEqual(response.status_code, 200)
        archive = zipfile.ZipFile(io.BytesIO(
            b''.join(response.streaming_content)))
        self.assertEqual(len(archive.namelist()), 3)

    def test_zip_temp_files_are_removed(self):
        response = self.client.get(BASE + 'download/', {
            'run': self.co.id, 'type': 'mid_artifacts', 'all_runs': '1'})
        b''.join(response.streaming_content)
        response.close()
        self.assertEqual(os.listdir(os.path.join(self.tmp, 'tmp')), [])

    def test_hash_of_a_file_from_the_old_upload_is_computed_once(self):
        StoredFile.objects.all().delete()
        with patch.object(file_util, 'sha256_of', wraps=file_util.sha256_of) as spy:
            self.list_files(self.pa1, 'mid_artifacts')
            self.list_files(self.pa1, 'mid_artifacts')
        self.assertEqual(spy.call_count, 1)
