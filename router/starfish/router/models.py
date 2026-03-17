import uuid

from django.db import models, transaction
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django_fsm import transition, FSMIntegerField


class Site(models.Model):
    """
    A site running local FL tasks
    """

    class SiteStatus(models.IntegerChoices):
        DISCONNECTED = 0
        CONNECTED = 1

    name = models.CharField(max_length=100, unique=True, blank=False)
    description = models.TextField()
    uid = models.UUIDField(primary_key=False, unique=True)
    owner = models.ForeignKey(
        'auth.User', default='admin', related_name='owner', on_delete=models.CASCADE)
    status = models.IntegerField(
        choices=SiteStatus.choices, default=SiteStatus.DISCONNECTED)
    created_at = models.DateTimeField(editable=False)
    updated_at = models.DateTimeField()

    def save(self, *args, **kwargs):
        """ On save, update timestamps """
        curr_time = timezone.now()
        if not self.id:
            self.created_at = curr_time
            self.status = Site.SiteStatus.CONNECTED
        self.updated_at = curr_time
        return super(Site, self).save(*args, **kwargs)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['id']


class Project(models.Model):
    """
        A project for a group of FL tasks defined.
    """
    name = models.CharField(max_length=100, blank=True, default='')
    description = models.TextField()
    site = models.ForeignKey(Site, on_delete=models.CASCADE)
    tasks = models.JSONField(encoder=None, decoder=None, default=[])
    batch = models.IntegerField()
    agent_config = models.JSONField(default=dict, blank=True)
    agent_log = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(editable=False)
    updated_at = models.DateTimeField()

    def save(self, *args, **kwargs):
        """ On save, update timestamps """
        curr_time = timezone.now()
        if not self.id:
            self.created_at = curr_time
            self.batch = 0
        self.updated_at = curr_time
        return super(Project, self).save(*args, **kwargs)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['id']


class ProjectParticipant(models.Model):
    """
    Participants of a project and their roles.
    """

    class Role(models.TextChoices):
        COORDINATOR = "CO", _("coordinator")
        PARTICIPANT = "PA", _("participant")

    site = models.ForeignKey(Site, on_delete=models.CASCADE)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    role = models.CharField(
        max_length=2,
        choices=Role.choices,
        default=Role.PARTICIPANT,
    )
    notes = models.TextField()
    created_at = models.DateTimeField(editable=False)
    updated_at = models.DateTimeField()

    def save(self, *args, **kwargs):
        """ On save, update timestamps """
        curr_time = timezone.now()
        if not self.id:
            self.created_at = curr_time
        self.updated_at = curr_time
        return super(ProjectParticipant, self).save(*args, **kwargs)

    def __str__(self):
        return self.project.name + '-' + self.site.name

    class Meta:
        ordering = ['id']
        unique_together = ('site', 'project',)


class Run(models.Model):
    """
    Runs of a project.
    """

    class RunStatus(models.IntegerChoices):
        FAILED = 0
        PENDING_FAILED = 1
        STANDBY = 2
        PREPARING = 3
        RUNNING = 4
        PENDING_SUCCESS = 5
        PENDING_AGGREGATING = 6
        AGGREGATING = 7
        SUCCESS = 8

    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    participant = models.ForeignKey(
        ProjectParticipant, on_delete=models.CASCADE)
    site_uid = models.UUIDField(default=uuid.uuid4())
    batch = models.IntegerField()
    cur_seq = models.IntegerField(default=1)
    tasks = models.JSONField(encoder=None, decoder=None, default=[])
    middle_artifacts = models.JSONField(
        encoder=None, decoder=None, default=[])
    role = models.CharField(
        max_length=2,
        choices=ProjectParticipant.Role.choices,
        default=ProjectParticipant.Role.PARTICIPANT,
    )
    status = FSMIntegerField(
        choices=RunStatus.choices, default=RunStatus.STANDBY, protected=True)
    logs = models.JSONField(encoder=None, decoder=None, default=[])
    artifacts = models.JSONField(encoder=None, decoder=None, default=[])
    agent_advice = models.JSONField(default=dict, blank=True)
    agent_diagnosis = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(editable=False)
    updated_at = models.DateTimeField()

    def save(self, *args, **kwargs):
        """ On save, update timestamps """
        curr_time = timezone.now()
        if not self.id:
            # copy the participant's role to the new record
            self.role = self.participant.role
            self.site_uid = self.participant.site.uid
            self.created_at = curr_time
            if ProjectParticipant.Role.COORDINATOR == self.role:
                self.project.batch += 1
                self.project.save()
            self.batch = self.project.batch
            self.cur_seq = 1
            self.tasks = self.project.tasks
            self.status = Run.RunStatus.STANDBY
        self.updated_at = curr_time
        return super(Run, self).save(*args, **kwargs)

    def __str__(self):
        return self.project.name + '-' + self.batch + '-' + self.id

    @staticmethod
    def update_status(instance, status):
        match status:
            case Run.RunStatus.STANDBY:
                instance.to_restart()
            case Run.RunStatus.PREPARING:
                instance.preparing()
            case Run.RunStatus.RUNNING:
                instance.running()
            case Run.RunStatus.PENDING_SUCCESS:
                instance.pending_success()
            case Run.RunStatus.PENDING_AGGREGATING:
                instance.pending_aggregating()
            case Run.RunStatus.AGGREGATING:
                instance.aggregating()
            case Run.RunStatus.PENDING_FAILED:
                instance.pending_failed()
            case Run.RunStatus.SUCCESS:
                instance.success()
            case Run.RunStatus.FAILED:
                instance.failed()
        return instance

    @transition(field=status, source="*", target=RunStatus.STANDBY)
    def to_restart(self):
        print(self.status)

    @transition(field=status,
                source=[RunStatus.STANDBY,
                        RunStatus.PREPARING, RunStatus.RUNNING],
                target=RunStatus.PENDING_FAILED)
    def to_stop(self):
        print(self.status)

    @transition(field=status, source=RunStatus.STANDBY, target=RunStatus.PREPARING)
    def preparing(self):
        print(self.status)

    @transition(field=status, source=RunStatus.PREPARING, target=RunStatus.RUNNING)
    def running(self):
        print(self.status)

    @transition(field=status, source=RunStatus.RUNNING, target=RunStatus.PENDING_SUCCESS)
    def pending_success(self):
        print(self.status)

    @transition(field=status, source=RunStatus.PENDING_SUCCESS, target=RunStatus.PENDING_AGGREGATING)
    def pending_aggregating(self):
        print(self.status)

    @transition(field=status, source=RunStatus.PENDING_AGGREGATING, target=RunStatus.AGGREGATING)
    def aggregating(self):
        print(self.status)

    @transition(field=status, source=[RunStatus.RUNNING, RunStatus.PREPARING], target=RunStatus.PENDING_FAILED)
    def pending_failed(self):
        print(self.status)

    @transition(field=status, source=[RunStatus.PENDING_SUCCESS, RunStatus.AGGREGATING], target=RunStatus.SUCCESS)
    def success(self):
        print(self.status)

    @transition(field=status, source=[RunStatus.PENDING_FAILED, RunStatus.PENDING_AGGREGATING, RunStatus.AGGREGATING], target=RunStatus.FAILED)
    def failed(self):
        print(self.status)

    class Meta:
        ordering = ['id']
        unique_together = ('project', 'participant', 'batch',)
