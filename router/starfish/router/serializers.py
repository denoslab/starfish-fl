from datetime import datetime

from django.contrib.auth.models import User, Group
from rest_framework import serializers
from starfish.router.models import Run

from rest_framework.validators import UniqueValidator

from starfish.router.models import Site, Project, ProjectParticipant
from django.db import transaction, DatabaseError


class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ['url', 'username', 'email', 'groups']


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ['url', 'name']


class SiteSerializer(serializers.ModelSerializer):
    id = serializers.IntegerField(read_only=True)
    name = serializers.CharField(
        required=True, allow_blank=False, max_length=100, validators=[UniqueValidator(queryset=Site.objects.all())])
    description = serializers.CharField(
        style={'base_template': 'textarea.html'})
    uid = serializers.UUIDField(format='hex_verbose', validators=[
                                UniqueValidator(queryset=Site.objects.all())])
    status = serializers.CharField(source='get_status_display', read_only=True)
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)

    def create(self, validated_data):
        """
        Create and return a new `Site` instance, given the validated data.
        """
        return Site.objects.create(**validated_data)

    def update(self, instance, validated_data):
        """
        Update and return an existing `Site` instance, given the validated data.
        """
        instance.name = validated_data.get('name', instance.name)
        instance.description = validated_data.get(
            'description', instance.description)
        instance.status = validated_data.get('status', instance.status)
        instance.updated_at = datetime.now()
        instance.save()
        return instance

    class Meta:
        model = Site
        fields = ['id', 'name', 'description', 'uid',
                  'status', 'created_at', 'updated_at']
        create_only_fields = ('uid',)


class TaskSerializer(serializers.Serializer):
    seq = serializers.IntegerField(required=True)
    model = serializers.CharField(required=True, allow_blank=False)
    config = serializers.JSONField(
        binary=False, default='{}', initial='{}', encoder=None)


class ProjectSerializer(serializers.ModelSerializer):
    id = serializers.IntegerField(read_only=True)
    name = serializers.CharField(
        required=True, allow_blank=False, max_length=100)
    description = serializers.CharField(
        style={'base_template': 'textarea.html'})
    site = serializers.PrimaryKeyRelatedField(
        many=False, queryset=Site.objects.all())
    batch = serializers.IntegerField(read_only=True)
    tasks = TaskSerializer(many=True)
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)

    def create(self, validated_data):
        """
        Create and return a new `Project` instance, given the validated data.
        """
        return Project.objects.create(**validated_data)

    def update(self, instance, validated_data):
        """
        Update and return an existing `Project` instance, given the validated data.
        """
        instance.name = validated_data.get('name', instance.name)
        instance.description = validated_data.get(
            'description', instance.description)
        instance.save()
        return instance

    def create_with_participant(self, validated_data):
        """
        check if project name exist, if not exist, will create a new project record current site is CO, otherwise is PA
        """

        project_name = validated_data.get("name")
        site_id = validated_data.get("site")
        description = validated_data.get("description")
        tasks = validated_data.get("tasks")

        project = Project.objects.filter(name=project_name).first()
        role = ProjectParticipant.Role.PARTICIPANT

        site = Site.objects.get(id=site_id)

        try:
            with transaction.atomic():
                if not project:
                    project = Project.objects.create(
                        site=site,
                        name=project_name,
                        description=description,
                        tasks=tasks
                    )
                    role = ProjectParticipant.Role.COORDINATOR
                ProjectParticipant.objects.get_or_create(
                    site=site, project=project, defaults={'site': site, 'project': project, 'role': role})
        except DatabaseError:
            raise serializers.ValidationError(
                'Encountered issue while creating project {}'.format(project_name))

    class Meta:
        model = Project
        fields = ['id', 'name', 'description', 'site', 'batch',
                  'tasks', 'agent_config', 'agent_log', 'created_at', 'updated_at']
        create_only_fields = ('site', 'tasks')


class ProjectParticipantSerializer(serializers.ModelSerializer):
    id = serializers.IntegerField(read_only=True)
    site = SiteSerializer(many=False)
    project = ProjectSerializer(many=False)
    role = serializers.CharField(source='get_role_display')
    notes = serializers.CharField(style={'base_template': 'textarea.html'})
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)

    def create(self, validated_data):
        """
        Create and return a new `ProjectParticipant` instance, given the validated data.
        """
        return ProjectParticipant.objects.create(**validated_data)

    def update(self, instance, validated_data):
        """
        Update and return an existing `ProjectParticipant` instance, given the validated data.
        """
        instance.notes = validated_data.get('notes', instance.notes)
        instance.save()
        return instance

    class Meta:
        model = ProjectParticipant
        fields = ['id', 'site', 'project', 'role',
                  'notes', 'created_at', 'updated_at']
        create_only_fields = ('site', 'project', 'role')


class ProjectParticipantCreateSerializer(ProjectParticipantSerializer):
    site = serializers.PrimaryKeyRelatedField(
        many=False, queryset=Site.objects.all())
    project = serializers.PrimaryKeyRelatedField(
        many=False, queryset=Project.objects.all())
    role = serializers.CharField()


class RunSerializer(serializers.ModelSerializer):
    id = serializers.IntegerField(read_only=True)
    project = serializers.PrimaryKeyRelatedField(
        many=False, queryset=Project.objects.all())
    batch = serializers.IntegerField(read_only=True)
    cur_seq = serializers.IntegerField(read_only=True)
    participant = serializers.PrimaryKeyRelatedField(
        many=False, queryset=ProjectParticipant.objects.all())
    role = serializers.CharField(source='get_role_display', read_only=True)
    site_uid = serializers.UUIDField(format='hex_verbose', read_only=True)
    status = serializers.CharField(source='get_status_display')
    logs = serializers.JSONField(
        binary=False, default='{}', initial='{}', encoder=None)
    tasks = TaskSerializer(many=True)
    middle_artifacts = serializers.JSONField(
        binary=False, default='{}', initial='{}', encoder=None)
    artifacts = serializers.JSONField(
        binary=False, default='{}', initial='{}', encoder=None)
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)

    def create(self, validated_data):
        """
        Create and return a new `Run` instance, given the validated data.
        """
        return Run.objects.create(**validated_data)

    def bulk_create(self, validated_data):
        return Run.objects.bulk_create([Run(**item) for item in validated_data], batch_size=100)

    def update(self, instance, validated_data):
        """
        Update and return an existing `Run` instance, given the validated data.
        """
        instance.logs = validated_data.get('logs', instance.logs)
        instance.artifacts = validated_data.get(
            'artifacts', instance.artifacts)
        instance.save()
        return instance

    class Meta:
        model = Run
        fields = ['id', 'project', 'batch', 'participant', 'role', 'site_uid', 'cur_seq',
                  'status', 'logs', 'artifacts', 'tasks', 'middle_artifacts',
                  'agent_advice', 'agent_diagnosis', 'created_at', 'updated_at']
        create_only_fields = ('project', 'participant', 'role')


class RunRetrieveSerializer(RunSerializer):
    project = ProjectSerializer()
    participant = ProjectParticipantSerializer()
