"""Tests for new agent-related model fields."""

from uuid import uuid4
from django.test import TestCase
from django.contrib.auth.models import User

from starfish.router.models import Site, Project, ProjectParticipant, Run


class TestProjectAgentFields(TestCase):
    """Test the new agent_config and agent_log fields on Project."""

    def setUp(self):
        self.user = User.objects.create_user(
            username='agenttest', password='testpass123')
        self.site = Site.objects.create(
            name='Agent Test Site', description='test',
            uid=uuid4(), owner=self.user)

    def test_project_agent_config_default_empty_dict(self):
        project = Project.objects.create(
            name='Test Project', description='test',
            site=self.site, batch=0)
        self.assertEqual(project.agent_config, {})

    def test_project_agent_log_default_empty_list(self):
        project = Project.objects.create(
            name='Test Project', description='test',
            site=self.site, batch=0)
        self.assertEqual(project.agent_log, [])

    def test_project_agent_config_can_be_set(self):
        project = Project.objects.create(
            name='Test Project', description='test',
            site=self.site, batch=0)
        project.agent_config = {"enabled": True, "aggregation": True}
        project.save()

        project.refresh_from_db()
        self.assertTrue(project.agent_config["enabled"])
        self.assertTrue(project.agent_config["aggregation"])

    def test_project_agent_log_can_append(self):
        project = Project.objects.create(
            name='Test Project', description='test',
            site=self.site, batch=0)
        project.agent_log.append({"event": "test", "data": "hello"})
        project.save()

        project.refresh_from_db()
        self.assertEqual(len(project.agent_log), 1)
        self.assertEqual(project.agent_log[0]["event"], "test")

    def test_project_without_agent_config_works(self):
        """Backward compatibility: projects without agent config function normally."""
        project = Project.objects.create(
            name='Legacy Project', description='no agent',
            site=self.site, batch=0)

        # These should all work without error
        self.assertEqual(project.agent_config, {})
        self.assertEqual(project.agent_log, [])
        self.assertFalse(project.agent_config.get("enabled", False))


class TestRunAgentFields(TestCase):
    """Test the new agent_advice and agent_diagnosis fields on Run."""

    def setUp(self):
        self.user = User.objects.create_user(
            username='agentruntest', password='testpass123')
        self.site = Site.objects.create(
            name='Run Agent Test Site', description='test',
            uid=uuid4(), owner=self.user)
        self.project = Project.objects.create(
            name='Run Agent Project', description='test',
            site=self.site, batch=0,
            tasks=[{"seq": 1, "model": "LogisticRegression",
                    "config": {"total_round": 2, "current_round": 1}}])
        self.participant = ProjectParticipant.objects.create(
            site=self.site, project=self.project,
            role=ProjectParticipant.Role.COORDINATOR,
            notes='test')

    def test_run_agent_advice_default_empty_dict(self):
        run = Run(project=self.project, participant=self.participant)
        run.save()
        self.assertEqual(run.agent_advice, {})

    def test_run_agent_diagnosis_default_empty_dict(self):
        run = Run(project=self.project, participant=self.participant)
        run.save()
        self.assertEqual(run.agent_diagnosis, {})

    def test_run_agent_advice_can_be_set(self):
        run = Run(project=self.project, participant=self.participant)
        run.save()
        run.agent_advice = {"action": "proceed", "reason": "all good"}
        run.save()

        run.refresh_from_db()
        self.assertEqual(run.agent_advice["action"], "proceed")

    def test_run_agent_diagnosis_can_be_set(self):
        run = Run(project=self.project, participant=self.participant)
        run.save()
        run.agent_diagnosis = {
            "root_cause": "missing data",
            "category": "data_quality",
            "severity": "recoverable",
            "suggestion": "fix it",
            "auto_action": None,
        }
        run.save()

        run.refresh_from_db()
        self.assertEqual(run.agent_diagnosis["category"], "data_quality")

    def test_run_serialization_includes_agent_fields(self):
        """Verify the RunSerializer includes the new fields."""
        from starfish.router.serializers import RunSerializer
        run = Run(project=self.project, participant=self.participant)
        run.save()

        serializer = RunSerializer(run)
        data = serializer.data
        self.assertIn("agent_advice", data)
        self.assertIn("agent_diagnosis", data)

    def test_project_serialization_includes_agent_fields(self):
        """Verify the ProjectSerializer includes the new fields."""
        from starfish.router.serializers import ProjectSerializer
        serializer = ProjectSerializer(self.project)
        data = serializer.data
        self.assertIn("agent_config", data)
        self.assertIn("agent_log", data)
