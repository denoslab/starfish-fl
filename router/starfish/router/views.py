from uuid import UUID

from django.contrib.auth.models import User, Group
from django.core.files.storage import FileSystemStorage
from django.db import transaction, DatabaseError
from django.http import HttpResponse
from django.utils import timezone
from rest_framework import permissions
from rest_framework import status
from rest_framework import viewsets, mixins, generics
from rest_framework.decorators import action
from rest_framework.generics import get_object_or_404
from rest_framework.response import Response
from rest_framework.viewsets import ViewSet

from starfish.router.models import Site, Project, ProjectParticipant, Run
from starfish.router.serializers import SiteSerializer, \
    ProjectSerializer, ProjectParticipantSerializer, \
    ProjectParticipantCreateSerializer, RunSerializer, \
    RunRetrieveSerializer
from starfish.router.serializers import UserSerializer, GroupSerializer
from starfish.utils import display_util
from ..utils.file_util import generate_url, get_file_urls, zip_all_files, gen_unique_file_name


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

    @action(detail=False, methods=['POST'], url_path='heartbeat')
    def heartbeat(self, request):
        """
        Sync heartbeat
        """

        uid_param = request.data.get('uid', None)
        status_param = request.data.get('status', None)

        if not validate_uuid4(uid_param):
            return Response("Invalid uid", status=status.HTTP_400_BAD_REQUEST)

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

    @action(detail=True, methods=['PUT'], url_path='status')
    def update_status(self, request, pk=None):
        run = self.get_object()
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
                runs.update(status=state)
            else:
                run = self.get_with_lock()
                run = Run.update_status(run, state)
                run.save()

        # Agent hooks (outside transaction to avoid blocking)
        self._run_agent_hooks(run, state, project_id)

        return Response(status=status.HTTP_202_ACCEPTED)

    def _run_agent_hooks(self, run, state, project_id):
        """Invoke agent hooks based on the new run state. Non-blocking."""
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
        if run:
            url = generate_url(run_id, task_seq, round_seq)
            if url:
                fs = FileSystemStorage(url)
                if artifacts_file:
                    runs = Run.objects.filter(batch=run.batch).exclude(id=run_id)
                    if runs:
                        for r in runs:
                            u = generate_url(r.id, task_seq, round_seq)
                            f = FileSystemStorage(u)
                            af = f.save(
                                gen_unique_file_name(artifacts_file.name, r.id, task_seq, round_seq),
                                artifacts_file)
                            if af:
                                r.artifacts.append(u + af)
                                r.save()
                    artifacts_file_name = fs.save(
                        gen_unique_file_name(artifacts_file.name, run_id, task_seq, round_seq), artifacts_file)
                    if not artifacts_file_name:
                        return Response("Error while saving artifacts", status=status.HTTP_400_BAD_REQUEST)
                    else:
                        run.artifacts.append(url + artifacts_file_name)

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
                    response = HttpResponse(
                        zip_file, content_type='application/zip')
                    response['Content-Disposition'] = f'attachment; filename="{file_type}.zip"'
                    return response
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
