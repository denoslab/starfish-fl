import os
import shutil
from uuid import UUID

from django.contrib.auth.models import User, Group
from django.core.files.storage import FileSystemStorage
from django.db import transaction, DatabaseError
from django.conf import settings
from django.http import FileResponse
from django.utils import timezone
from rest_framework import permissions
from rest_framework import status
from rest_framework import viewsets, mixins, generics
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied
from rest_framework.generics import get_object_or_404
from rest_framework.response import Response
from rest_framework.viewsets import ViewSet

from starfish.router.auth import check_run_access, check_site_uid, hash_secret, new_secret, \
    request_site
from starfish.router.models import Site, Project, ProjectParticipant, Run, StoredFile, ModelVersion, \
    EnrolmentCode, SiteToken
from starfish.router.serializers import SiteSerializer, \
    ProjectSerializer, ProjectParticipantSerializer, \
    ProjectParticipantCreateSerializer, RunSerializer, \
    RunRetrieveSerializer
from starfish.router.serializers import UserSerializer, GroupSerializer
from starfish.utils import display_util
from ..utils import file_util
from ..utils.file_util import generate_url, get_file_urls, zip_all_files, gen_unique_file_name, \
    gen_batch_url, TransferError


def validate_uuid4(uuid_string):
    """
    Validate that a UUID string is in
    fact a valid uuid4.
    Happily, the uuid module does the actual
    checking for us.
    It is vital that the 'version' kwarg be passed
    to the UUID() call, otherwise any 32-character
    hex string is considered valid.
    """

    try:
        val = UUID(uuid_string, version=4)
    except ValueError:
        # If it's a value error, then the string
        # is not a valid hex code for a UUID.
        return False

    return True


class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    permission_classes = [permissions.IsAuthenticated]


class SiteViewSet(viewsets.ModelViewSet):
    """
    This viewset automatically provides `list`, `create`, `retrieve`,
    `update` and `destroy` actions.
    """
    queryset = Site.objects.all()
    serializer_class = SiteSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)

    @action(detail=False, methods=['GET'], url_path='lookup')
    def lookup_sites_by_uid(self, request):
        """
        Look up a site by its uid.
        """
        uid_param = request.GET.get('uid', None)
        if not validate_uuid4(uid_param):
            return Response("Invalid uid", status=status.HTTP_400_BAD_REQUEST)
        try:
            queryset = Site.objects.get(uid=uid_param)
        except Site.DoesNotExist:
            return Response("Site not found", status=status.HTTP_404_NOT_FOUND)
        serializer = SiteSerializer(queryset)
        return Response(serializer.data)

    @action(detail=False, methods=['POST'], url_path='enrol',
            permission_classes=[permissions.AllowAny], authentication_classes=[])
    def enrol(self, request):
        """
        Enrol a site with a single-use code and return its token once, SF-09.

        Body: code, uid, name, description. The token is not stored and
        cannot be shown again.
        """
        code = request.data.get('code')
        uid = request.data.get('uid')
        name = request.data.get('name')
        if not code or not validate_uuid4(uid) or not name:
            return Response("code, a uuid4 uid and name are required",
                            status=status.HTTP_400_BAD_REQUEST)
        with transaction.atomic():
            record = EnrolmentCode.objects.select_for_update().filter(
                code_hash=hash_secret(str(code))).first()
            if record is None or record.used_at is not None or record.expires_at <= timezone.now():
                return Response("Invalid, used or expired enrolment code",
                                status=status.HTTP_403_FORBIDDEN)
            site = Site.objects.filter(uid=uid).first()
            if site is None:
                owner = User.objects.filter(
                    is_superuser=True).order_by('id').first()
                site = Site.objects.create(uid=uid, name=name, owner=owner,
                                           description=request.data.get('description') or '')
            token = new_secret()
            SiteToken.objects.create(site=site, token_hash=hash_secret(token))
            record.used_at = timezone.now()
            record.used_by = site
            record.save()
            if record.project_id:
                ProjectParticipant.objects.get_or_create(
                    site=site, project_id=record.project_id,
                    defaults={'role': ProjectParticipant.Role.PARTICIPANT, 'notes': 'enrolled'})
        return Response({'site': site.id, 'uid': str(site.uid), 'token': token,
                         'project': record.project_id}, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['POST'], url_path='heartbeat')
    def heartbeat(self, request):
        """
        Sync heartbeat
        """

        uid_param = request.data.get('uid', None)
        status_param = request.data.get('status', None)

        if not validate_uuid4(uid_param):
            return Response("Invalid uid", status=status.HTTP_400_BAD_REQUEST)
        check_site_uid(request, uid_param)

        if not status_param in Site.SiteStatus:
            return Response("Status not supported", status=status.HTTP_400_BAD_REQUEST)

        try:
            with transaction.atomic():
                site = Site.objects.select_for_update().get(uid=uid_param)
                site.status = status_param
                site.save()
            return Response(status=status.HTTP_202_ACCEPTED)
        except DatabaseError:
            return Response(status=status.HTTP_422_UNPROCESSABLE_ENTITY)


class ProjectViewSet(viewsets.ModelViewSet):
    """
    This viewset automatically provides `list`, `create`, `retrieve`,
    `update` and `destroy` actions.
    """
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer
    permission_classes = [permissions.IsAuthenticated]

    def create(self, request, *args, **kwargs):
        site = request_site(request)
        if site is not None and str(request.data.get('site')) != str(site.id):
            raise PermissionDenied(
                'a site can create or join projects only as itself')
        serializer = self.serializer_class(data=request.data, partial=True)

        if serializer.is_valid():
            serializer.create_with_participant(request.data)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['GET'], url_path='lookup')
    def lookup_projects_by_site_id(self, request):
        """
        Look up ProjectParticipant by site ID/name.
        All projects this site is involved will be returned.
        """
        site_id_param = request.GET.get('site_id', None)
        name_param = request.GET.get('name', None)
        if site_id_param:
            try:
                queryset = ProjectParticipant.objects.filter(
                    site=site_id_param)
            except ProjectParticipant.DoesNotExist:
                return Response("ProjectParticipant not found", status=status.HTTP_404_NOT_FOUND)
            serializer = ProjectParticipantSerializer(queryset, many=True)
        else:
            try:
                queryset = Project.objects.get(name=name_param)
            except Project.DoesNotExist:
                return Response("Project not found", status=status.HTTP_404_NOT_FOUND)
            serializer = ProjectSerializer(queryset, many=False)
        return Response(serializer.data)


class ProjectParticipantViewSet(viewsets.ModelViewSet):
    """
    This viewset automatically provides `list`, `create`, `retrieve`,
    `update` and `destroy` actions.
    """
    queryset = ProjectParticipant.objects.all()
    serializer_class = ProjectParticipantSerializer
    create_serializer_class = ProjectParticipantCreateSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_serializer_class(self):
        if self.action == 'create':
            if hasattr(self, 'create_serializer_class'):
                return self.create_serializer_class
        return super(ProjectParticipantViewSet, self).get_serializer_class()

    def perform_create(self, serializer):
        site = request_site(self.request)
        if site is not None and str(self.request.data.get('site')) != str(site.id):
            raise PermissionDenied('a site can join projects only as itself')
        serializer.save()

    @action(detail=False, methods=['GET'], url_path='lookup')
    def get_participants_by_project(self, request):
        """
        Look up participants by project id.
        """
        project_id = request.GET.get('project', None)
        queryset = ProjectParticipant.objects.filter(project_id=project_id)
        participants_data = ProjectParticipantSerializer(
            queryset, many=True).data
        return Response(participants_data)


class RunViewSet(mixins.RetrieveModelMixin, mixins.UpdateModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    """
    This viewset automatically provides `list`, `create`, `retrieve`,
    `update` and `destroy` actions.
    """
    queryset = Run.objects.all()
    serializer_class = RunSerializer
    retrieve_serializer_class = RunRetrieveSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_serializer_class(self):
        if self.action == 'retrieve':
            if hasattr(self, 'retrieve_serializer_class'):
                return self.retrieve_serializer_class
        return super(RunViewSet, self).get_serializer_class()

    def update(self, request, *args, **kwargs):
        instance = self.get_object()
        data = {
            "log": request.data.get('log', None),
            "artifacts": request.data.get('artifacts', None),
        }
        serializer = self.serializer_class(
            instance=instance, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_202_ACCEPTED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get_queryset(self):
        """A token site sees only its own runs through the standard routes, SF-09."""
        queryset = super().get_queryset()
        site = request_site(self.request)
        if site is not None:
            queryset = queryset.filter(site_uid=site.uid)
        return queryset

    @action(detail=True, methods=['PUT'], url_path='status')
    def update_status(self, request, pk=None):
        run = self.get_object()
        check_run_access(request, run)
        state = request.data.get('status', None)
        increase_round = request.data.get('increase_round', False)
        update_all = request.data.get('update_all', False)
        project_id = run.project.id
        if not run:
            return Response("Run not found", status=status.HTTP_400_BAD_REQUEST)
        if state is None:
            return Response("status is invalid", status=status.HTTP_400_BAD_REQUEST)

        with transaction.atomic():
            project = Project.objects.select_for_update().get(id=project_id)
            if not project:
                return Response("Failed to get project of run {}".format(run.id),
                                status=status.HTTP_400_BAD_REQUEST)
            if run.role == ProjectParticipant.Role.COORDINATOR and update_all:
                runs = Run.objects.select_for_update().filter(
                    project=project_id, batch=run.batch)
                # Partial participation, SF-10: runs to mark as sitting out this round
                sit_out = [int(i) for i in (request.data.get('sit_out') or [])
                           if int(i) != run.id]
                if increase_round:
                    if run.cur_seq <= len(run.tasks):
                        tasks = run.tasks
                        task = tasks[run.cur_seq - 1]
                        if task['config']['current_round'] < task['config']['total_round']:
                            task['config']['current_round'] += 1
                            tasks[run.cur_seq - 1] = task
                            runs.update(tasks=tasks)
                        else:
                            if run.cur_seq < len(run.tasks):
                                runs.update(cur_seq=run.cur_seq + 1)
                now = timezone.now()
                if int(state) == Run.RunStatus.STANDBY:
                    # A new round: every site takes part again
                    runs.update(status=state, updated_at=now)
                else:
                    runs.exclude(status=Run.RunStatus.SITTING_OUT).exclude(
                        id__in=sit_out).update(status=state, updated_at=now)
                    if sit_out:
                        runs.filter(id__in=sit_out).update(
                            status=Run.RunStatus.SITTING_OUT, updated_at=now)
            else:
                run = self.get_with_lock()
                run = Run.update_status(run, state)
                run.save()

        # Agent hooks (outside transaction to avoid blocking)
        self._run_agent_hooks(run, state, project_id)

        return Response(status=status.HTTP_202_ACCEPTED)

    def _run_agent_hooks(self, run, state, project_id):
        """Invoke agent hooks based on the new run state. Non-blocking.

        Skipped entirely, without importing agent code, when the router's
        environment sets ``STARFISH_DISABLE_AGENTS``.
        """
        if os.getenv('STARFISH_DISABLE_AGENTS', '').strip().lower() in ('1', 'true', 'yes'):
            return
        try:
            from starfish.agent import hooks as agent_hooks

            if state == Run.RunStatus.AGGREGATING:
                all_batch_runs = Run.objects.filter(
                    project_id=project_id, batch=run.batch)
                agent_hooks.on_aggregating(run, all_batch_runs)

            elif state == Run.RunStatus.SUCCESS:
                if run.role == ProjectParticipant.Role.COORDINATOR:
                    agent_hooks.on_success(run)

            elif state in (Run.RunStatus.FAILED, Run.RunStatus.PENDING_FAILED):
                agent_hooks.on_failed(run)
        except Exception:
            # Agent hooks must never break state transitions
            import logging
            logging.getLogger(__name__).warning(
                "Agent hook failed for run %s state %s", run.id, state,
                exc_info=True)

    def get_with_lock(self, queryset=None):
        # Acquire an exclusive lock on the object using select_for_update()
        return get_object_or_404(Run.objects.select_for_update(), pk=self.kwargs['pk'])

    @action(detail=False, methods=['GET'], url_path='lookup')
    def lookup_runs_by_project_id(self, request):
        """
        Look up runs by project id.
        """
        site_uid = request.GET.get('site_uid', None)
        project_id = request.GET.get('project', None)
        batch_id = request.GET.get('batch_id', None)
        if batch_id:
            queryset = Run.objects.filter(
                project_id=project_id, batch=batch_id)
        else:
            queryset = Run.objects.filter(project_id=project_id)
        serializer = RunSerializer(queryset, many=True)
        dic = display_util.sort_runs(serializer.data, site_uid=site_uid)
        return Response(dic)

    @action(detail=False, methods=['GET'], url_path='active')
    def get_active_runs(self, request):
        queryset = Run.objects.exclude(
            status__in=[Run.RunStatus.FAILED, Run.RunStatus.SUCCESS])
        if request_site(request) is not None:
            queryset = queryset.filter(site_uid=request_site(request).uid)
        serializer = RunSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    @action(detail=False, methods=['GET'], url_path='detail')
    def get_runs_details(self, request):
        """
        Get runs details by batch , project_id and site_id.
        """
        batch = request.GET.get('batch', None)
        project_id = request.GET.get('project', None)
        site = request.GET.get('site', None)
        site_uid = request.GET.get('site_uid', None)
        token_site = request_site(request)
        if token_site is not None:
            if site_uid is None and str(site) != str(token_site.id):
                raise PermissionDenied('a site can look up only its own runs')
            check_site_uid(request, site_uid or token_site.uid)

        if site_uid:
            site_id = Site.objects.get(uid=site_uid)
            participant_queryset = ProjectParticipant.objects.get(
                project_id=project_id, site_id=site_id.id)
        else:
            participant_queryset = ProjectParticipant.objects.get(
                project_id=project_id, site_id=site)
        participant_serializer = ProjectParticipantSerializer(
            participant_queryset)
        participant_data = participant_serializer.data
        role = None
        participant_id = None
        if participant_data:
            role = participant_data['role']
            participant_id = participant_data['id']
        run_queryset = Run.objects.filter(project_id=project_id, batch=batch)
        run_serializer = RunSerializer(run_queryset, many=True)
        run_data = run_serializer.data
        dic = display_util.pick_runs(run_data, role, participant_id)
        return Response(dic)


class BulkCreateRunAPIView(generics.ListCreateAPIView):
    # serializer_class = RunSerializer
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        project_id = request.data.get('project', None)
        site = request_site(request)
        if site is not None and not Project.objects.filter(id=project_id, site_id=site.id).exists():
            raise PermissionDenied(
                'only the project coordinator can start runs')
        queryset = Run.objects.filter(project_id=project_id)
        serializer = RunSerializer(queryset, many=True)
        should_create_new_runs = display_util.should_create_new_runs(
            serializer.data)
        if not should_create_new_runs:
            return Response("Last round of runs not completed", status=status.HTTP_400_BAD_REQUEST)
        project = Project.objects.get(id=project_id)
        if project_id and project:
            curr_time = timezone.now()
            project.batch += 1
            project.save()
            records_to_create = []
            pps = ProjectParticipant.objects.filter(project=project_id)
            for pp in pps:
                if pp.site.status == 1:
                    data = {
                        "project": project,
                        "participant": pp,
                        "site_uid": pp.site.uid,
                        "role": pp.role,
                        "status": Run.RunStatus.STANDBY,
                        "tasks": project.tasks,
                        "batch": project.batch,
                        "cur_seq": 1,
                        "created_at": curr_time,
                        "updated_at": curr_time
                    }
                    records_to_create.append(data)
            if len(records_to_create) == len(pps):
                created_records = Run.objects.bulk_create(
                    [Run(**item) for item in records_to_create], batch_size=100)
                if created_records:
                    return Response(status=status.HTTP_201_CREATED)
                else:
                    return Response("Error while creating runs", status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response("Not all sites are connected", status=status.HTTP_400_BAD_REQUEST)
        return Response("project not found", status=status.HTTP_400_BAD_REQUEST)


class RunsActionViewSet(ViewSet):
    """
    This method is used by fl tasks to upload its artifacts and logs from local volume upon runs' task and round success
    """
    permission_classes = [permissions.IsAuthenticated]

    RUN_FILE_FIELDS = {'artifacts': 'artifacts',
                       'mid_artifacts': 'middle_artifacts', 'logs': 'logs'}

    @staticmethod
    def _batch_runs(run):
        return Run.objects.filter(project_id=run.project_id, batch=run.batch)

    def _record(self, run, file_type, path):
        """Add a stored file to a run; an aggregated artifact to every run of the batch."""
        field = self.RUN_FILE_FIELDS[file_type]
        runs = self._batch_runs(run) if file_type == 'artifacts' else [run]
        for r in runs:
            paths = getattr(r, field)
            if path not in paths:
                paths.append(path)
                r.save()

    def _visible_paths(self, request):
        """Paths the caller may list or fetch, from run, type, task_seq, round_seq, all_runs."""
        run_id = request.GET.get('run')
        file_type = request.GET.get('type')
        if not run_id or file_type not in file_util.FILE_TYPES:
            raise TransferError('run and a valid type are required')
        run = Run.objects.filter(id=run_id).first()
        if run is None:
            raise TransferError('run not found', status=404)
        check_run_access(request, run)
        runs = [run]
        if request.GET.get('all_runs', '0') == '1' and run.role == ProjectParticipant.Role.COORDINATOR:
            runs = list(self._batch_runs(run))
        return get_file_urls(runs, request.GET.get('task_seq'), request.GET.get('round_seq'), file_type)

    @staticmethod
    def _describe(path):
        """Size and SHA-256 of a stored file, hashing it once if it predates StoredFile."""
        stored = StoredFile.objects.filter(path=path).first()
        size = os.path.getsize(path)
        if stored is None or stored.size != size:
            stored, _ = StoredFile.objects.update_or_create(
                path=path, defaults={'size': size, 'sha256': file_util.sha256_of(path)})
        return {'name': os.path.basename(path), 'size': stored.size, 'sha256': stored.sha256}

    @action(detail=False, methods=['GET'], url_path='files')
    def list_files(self, request):
        """List files with name, size and SHA-256, instead of zipping them."""
        try:
            paths = self._visible_paths(request)
        except TransferError as e:
            return Response(str(e), status=e.status)
        return Response([self._describe(p) for p in paths if os.path.isfile(p)])

    @action(detail=False, methods=['GET', 'PUT'], url_path='file')
    def file(self, request):
        """GET streams one listed file by name. PUT receives one file as a streamed body."""
        if request.method == 'PUT':
            return self._receive_file(request)
        name = request.GET.get('name')
        try:
            paths = self._visible_paths(request)
        except TransferError as e:
            return Response(str(e), status=e.status)
        for path in paths:
            if os.path.basename(path) == name and os.path.isfile(path):
                info = self._describe(path)
                response = FileResponse(
                    open(path, 'rb'), as_attachment=True, filename=name)
                response['X-Starfish-SHA256'] = info['sha256']
                return response
        return Response('file not found', status=status.HTTP_404_NOT_FOUND)

    def _receive_file(self, request):
        params = request.GET
        try:
            task_seq, round_seq = int(params.get('task_seq')), int(
                params.get('round_seq'))
        except (TypeError, ValueError):
            return Response('task_seq and round_seq must be integers', status=status.HTTP_400_BAD_REQUEST)
        file_type, name, sha256 = params.get(
            'type'), params.get('name'), params.get('sha256')
        try:
            size = file_util.check_upload_params(
                file_type, name, params.get('size'), sha256, settings.STARFISH_MAX_ARTIFACT_BYTES)
            length = request.META.get('CONTENT_LENGTH')
            if length not in (None, '') and int(length) != size:
                raise TransferError(
                    'Content-Length {} does not match size {}'.format(length, size))
            run = Run.objects.filter(id=params.get('run')).first()
            if run is None:
                raise TransferError('run not found', status=404)
            check_run_access(request, run)
            file_util.check_free_disk(
                size, settings.STARFISH_ARTIFACT_DISK_RESERVE_BYTES, shutil.disk_usage)
            if file_type == 'artifacts':
                dest = gen_batch_url(
                    run.project_id, run.batch, task_seq, round_seq)
            else:
                dest = generate_url(run.id, task_seq, round_seq)
            final_name = gen_unique_file_name(
                name, run.id, task_seq, round_seq)
            # Read the raw body in chunks; request.data would load it into memory
            path = file_util.receive_stream(
                request._request.read, dest, final_name, size, sha256)
        except TransferError as e:
            return Response(str(e), status=e.status)
        StoredFile.objects.update_or_create(
            path=path, defaults={'size': size, 'sha256': sha256})
        self._record(run, file_type, path)
        return Response({'name': final_name, 'size': size, 'sha256': sha256},
                        status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['POST'], url_path='upload')
    def upload(self, request):

        artifacts_file = request.FILES.get('artifacts')
        logs_file = request.FILES.get('logs')
        mid_artifacts_file = request.FILES.get('mid_artifacts')

        run_id = request.POST.get('run', None)
        task_seq = request.POST.get('task_seq', None)
        round_seq = request.POST.get('round_seq', None)

        if not run_id or not task_seq or not round_seq:
            return Response("Invalid uploaded  params", status=status.HTTP_400_BAD_REQUEST)

        if not artifacts_file and not logs_file and not mid_artifacts_file:
            return Response("Must at least upload one file", status=status.HTTP_400_BAD_REQUEST)

        run = Run.objects.get(id=run_id)
        check_run_access(request, run)
        if run:
            url = generate_url(run_id, task_seq, round_seq)
            if url:
                fs = FileSystemStorage(url)
                if artifacts_file:
                    # Stored once per batch and round, and listed on every run of the batch
                    batch_url = gen_batch_url(
                        run.project_id, run.batch, task_seq, round_seq)
                    artifacts_file_name = FileSystemStorage(batch_url).save(
                        gen_unique_file_name(artifacts_file.name, run_id, task_seq, round_seq), artifacts_file)
                    if not artifacts_file_name:
                        return Response("Error while saving artifacts", status=status.HTTP_400_BAD_REQUEST)
                    for r in self._batch_runs(run).exclude(id=run.id):
                        r.artifacts.append(batch_url + artifacts_file_name)
                        r.save()
                    run.artifacts.append(batch_url + artifacts_file_name)

                if logs_file:
                    logs_file_name = fs.save(gen_unique_file_name(
                        logs_file.name, run_id, task_seq, round_seq), logs_file)
                    if not logs_file_name:
                        return Response("Error while saving logs", status=status.HTTP_400_BAD_REQUEST)
                    else:
                        run.logs.append(url + logs_file_name)
                if mid_artifacts_file:
                    mid_artifacts_file_name = fs.save(
                        gen_unique_file_name(mid_artifacts_file.name, run_id, task_seq, round_seq), mid_artifacts_file)
                    if not mid_artifacts_file_name:
                        return Response("Error while saving mid-artifacts", status=status.HTTP_400_BAD_REQUEST)
                    else:
                        run.middle_artifacts.append(url + mid_artifacts_file_name)
                run.save()
                return Response(status=status.HTTP_200_OK)
        return Response("No run found", status=status.HTTP_400_BAD_REQUEST)

    """
    This method used to download artifacts or logs of run(s) including all tasks and inner rounds
    """

    @action(detail=False, methods=['GET'], url_path='download')
    def download(self, request):
        run_id = request.GET.get('run', None)
        all_runs = request.GET.get('all_runs', '0')
        file_type = request.GET.get('type', None)
        task_seq = request.GET.get('task_seq', None)
        round_seq = request.GET.get('round_seq', None)

        if not run_id or not file_type:
            return Response("Run id or file type not provided", status=status.HTTP_400_BAD_REQUEST)
        run = Run.objects.get(id=run_id)
        check_run_access(request, run)

        if run:
            runs = []
            if run.role == 'CO' and all_runs == '1':
                project_id = run.project_id
                batch = run.batch
                all_runs = Run.objects.filter(
                    project_id=project_id, batch=batch)
                runs.extend(all_runs)
            else:
                runs.append(run)

            urls = get_file_urls(runs, task_seq, round_seq, file_type)

            if urls and len(urls) > 0:
                zip_file = zip_all_files(run, urls, file_type)
                if zip_file:
                    return FileResponse(zip_file, as_attachment=True, filename=f'{file_type}.zip',
                                        content_type='application/zip')
            return Response("No files of {} found".format(file_type), status=status.HTTP_404_NOT_FOUND)
        return Response("Run not exist", status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['PUT'], url_path='update')
    def update_status_by_action(self, request, pk=None):
        run_id = request.data.get('run', None)
        action_role = request.data.get('role', None)
        request_action = request.data.get('action', None)
        project_id = request.data.get('project', None)
        batch = request.data.get('batch', None)

        if not run_id or not action_role or not request_action or not project_id or not batch:
            return Response("Run info absent", status=status.HTTP_400_BAD_REQUEST)
        requester = Run.objects.filter(id=run_id).first()
        if requester is None:
            return Response("Run not found", status=status.HTTP_404_NOT_FOUND)
        check_run_access(request, requester)
        if request_site(request) is not None and (
                str(requester.project_id) != str(project_id) or str(requester.batch) != str(batch)):
            raise PermissionDenied(
                'the run does not belong to that project and batch')

        target_status = display_util.get_status_from_action(request_action)
        if not target_status:
            return Response("Failed to get target status from action {}".format(request_action),
                            status=status.HTTP_400_BAD_REQUEST)

        if action_role == 'coordinator':
            with transaction.atomic():
                project = Project.objects.select_for_update().get(id=project_id)
                if not project:
                    return Response("Failed to get project of run {}".format(run_id),
                                    status=status.HTTP_400_BAD_REQUEST)
                run = Run.objects.select_for_update().filter(project=project_id, batch=batch).exclude(
                    status=target_status)
                if not run:
                    return Response("Failed to get run could perform action {}".format(request_action),
                                    status=status.HTTP_400_BAD_REQUEST)
                run.update(status=target_status)
                return Response(
                    "Update runs of project {} in batch {}  status to {}".format(
                        project_id, batch, target_status),
                    status=status.HTTP_202_ACCEPTED)
        else:
            with transaction.atomic():
                project = Project.objects.select_for_update().get(id=project_id)
                if not project:
                    return Response("Failed to get project of run {}".format(run_id),
                                    status=status.HTTP_400_BAD_REQUEST)
                run = Run.objects.select_for_update().get(id=run_id)
                if not run or run.status == target_status:
                    return Response("Failed to get run could perform action {}".format(request_action),
                                    status=status.HTTP_400_BAD_REQUEST)
                if target_status == 1:
                    run.to_stop()
                    run.save()
                else:
                    run.to_restart()
                    run.save()
                return Response("Update run {} status to {}".format(run_id, target_status),
                                status=status.HTTP_202_ACCEPTED)


class ModelRegistryViewSet(ViewSet):
    """
    Approved global models per frequency bucket, SF-12.

    - ``POST registry/`` publishes a run's aggregated artifact: run, task_seq,
      round_seq, bucket_hz, source_version, optional parent and eval_report.
      Only the coordinator's run can publish.
    - ``GET registry/?bucket_hz=`` lists approved versions, newest first.
    - ``GET registry/latest/?bucket_hz=`` gives the newest one.
    - ``GET registry/file/?version=`` streams its file with ``X-Starfish-SHA256``.
    """
    permission_classes = [permissions.IsAuthenticated]

    @staticmethod
    def _describe(mv):
        return {'version': mv.version, 'bucket_hz': mv.bucket_hz, 'sha256': mv.sha256,
                'size': mv.size, 'parent': mv.parent.version if mv.parent else None,
                'source_version': mv.source_version, 'project': mv.project_id,
                'batch': mv.batch, 'task_seq': mv.task_seq, 'round_seq': mv.round_seq,
                'eval_report': mv.eval_report, 'created_at': mv.created_at}

    @staticmethod
    def _bucket(request):
        try:
            return int(request.GET.get('bucket_hz'))
        except (TypeError, ValueError):
            raise TransferError('bucket_hz must be an integer')

    def list(self, request):
        try:
            bucket = self._bucket(request)
        except TransferError as e:
            return Response(str(e), status=e.status)
        return Response([self._describe(mv) for mv in ModelVersion.objects.filter(bucket_hz=bucket)])

    @action(detail=False, methods=['GET'], url_path='latest')
    def latest(self, request):
        try:
            bucket = self._bucket(request)
        except TransferError as e:
            return Response(str(e), status=e.status)
        mv = ModelVersion.objects.filter(bucket_hz=bucket).first()
        if mv is None:
            return Response('no approved model for this bucket', status=status.HTTP_404_NOT_FOUND)
        return Response(self._describe(mv))

    @action(detail=False, methods=['GET'], url_path='file')
    def file(self, request):
        mv = ModelVersion.objects.filter(
            version=request.GET.get('version')).first()
        if mv is None or not os.path.isfile(mv.path):
            return Response('model version not found', status=status.HTTP_404_NOT_FOUND)
        response = FileResponse(open(mv.path, 'rb'), as_attachment=True,
                                filename='{}.safetensors'.format(mv.version))
        response['X-Starfish-SHA256'] = mv.sha256
        return response

    def create(self, request):
        data = request.data
        try:
            run = Run.objects.filter(id=data.get('run')).first()
            if run is None:
                raise TransferError('run not found', status=404)
            check_run_access(request, run)
            if run.role != ProjectParticipant.Role.COORDINATOR:
                raise TransferError(
                    'only the coordinator can publish a model', status=403)
            task_seq, round_seq = int(
                data.get('task_seq')), int(data.get('round_seq'))
            bucket = int(data.get('bucket_hz'))
            source_version = str(data.get('source_version') or '')
            if not source_version:
                raise TransferError('source_version is required')
            paths = get_file_urls([run], task_seq, round_seq, 'artifacts')
            if len(paths) != 1 or not os.path.isfile(paths[0]):
                raise TransferError('expected one aggregated artifact for that round, found {}'.format(
                    len(paths)), status=404)
            parent = None
            if data.get('parent'):
                parent = ModelVersion.objects.filter(version=data.get('parent'),
                                                     bucket_hz=bucket).first()
                if parent is None:
                    raise TransferError('parent version not found', status=404)
        except (TypeError, ValueError):
            return Response('task_seq, round_seq and bucket_hz must be integers',
                            status=status.HTTP_400_BAD_REQUEST)
        except TransferError as e:
            return Response(str(e), status=e.status)

        existing = ModelVersion.objects.filter(project_id=run.project_id, batch=run.batch,
                                               task_seq=task_seq, round_seq=round_seq).first()
        if existing is not None:
            return Response(self._describe(existing), status=status.HTTP_200_OK)
        source = paths[0]
        info = RunsActionViewSet._describe(source)
        with transaction.atomic():
            last = ModelVersion.objects.select_for_update().filter(bucket_hz=bucket).first()
            sequence = (last.sequence + 1) if last else 1
            version = '{}k-v{:04d}'.format(bucket // 1000, sequence)
            folder = os.path.join(file_util.base_folder, 'registry')
            os.makedirs(folder, exist_ok=True)
            target = os.path.join(folder, version + '.safetensors')
            shutil.copyfile(source, target)
            if file_util.sha256_of(target) != info['sha256']:
                os.remove(target)
                return Response('copy of the artifact does not match its hash',
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            report = data.get('eval_report') or {}
            mv = ModelVersion.objects.create(
                version=version, bucket_hz=bucket, sequence=sequence, path=target,
                size=info['size'], sha256=info['sha256'], parent=parent, project_id=run.project_id,
                batch=run.batch, task_seq=task_seq, round_seq=round_seq,
                source_version=source_version, eval_report=report if isinstance(report, dict) else {})
        return Response(self._describe(mv), status=status.HTTP_201_CREATED)


class EnrolmentCodeViewSet(ViewSet):
    """
    Admin only, SF-09. ``POST enrolment-codes/`` with optional project, note and
    ``valid_hours``, default 72, returns a single-use code once. ``GET`` lists
    codes without the codes themselves.
    """
    permission_classes = [permissions.IsAdminUser]

    def list(self, request):
        return Response([{'id': c.id, 'project': c.project_id, 'note': c.note,
                          'expires_at': c.expires_at, 'used_at': c.used_at,
                          'used_by': c.used_by_id} for c in EnrolmentCode.objects.order_by('-id')])

    def create(self, request):
        try:
            hours = float(request.data.get('valid_hours', 72))
        except (TypeError, ValueError):
            return Response('valid_hours must be a number', status=status.HTTP_400_BAD_REQUEST)
        project = None
        if request.data.get('project'):
            project = Project.objects.filter(
                id=request.data.get('project')).first()
            if project is None:
                return Response('project not found', status=status.HTTP_404_NOT_FOUND)
        code = new_secret()
        record = EnrolmentCode.objects.create(
            code_hash=hash_secret(code), project=project, note=str(request.data.get('note') or '')[:200],
            expires_at=timezone.now() + timezone.timedelta(hours=hours))
        return Response({'id': record.id, 'code': code, 'project': record.project_id,
                         'expires_at': record.expires_at}, status=status.HTTP_201_CREATED)


class SiteTokenViewSet(ViewSet):
    """Admin only, SF-09. ``GET site-tokens/`` lists tokens; ``POST site-tokens/<id>/revoke/``."""
    permission_classes = [permissions.IsAdminUser]

    def list(self, request):
        return Response([{'id': t.id, 'site': t.site_id, 'site_uid': str(t.site.uid),
                          'created_at': t.created_at, 'last_used_at': t.last_used_at,
                          'revoked_at': t.revoked_at}
                         for t in SiteToken.objects.select_related('site').order_by('-id')])

    @action(detail=True, methods=['POST'], url_path='revoke')
    def revoke(self, request, pk=None):
        token = SiteToken.objects.filter(pk=pk).first()
        if token is None:
            return Response('token not found', status=status.HTTP_404_NOT_FOUND)
        if token.revoked_at is None:
            token.revoked_at = timezone.now()
            token.save()
        return Response({'id': token.id, 'revoked_at': token.revoked_at})
