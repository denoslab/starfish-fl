"""
Tests for the system prompt content.
"""

from starfish_cli.agent.prompts import SYSTEM_PROMPT


class TestSystemPrompt:
    """Verify the system prompt contains essential FL domain knowledge."""

    def test_prompt_is_non_empty(self):
        assert len(SYSTEM_PROMPT) > 100

    def test_prompt_mentions_starfish(self):
        assert "Starfish-FL" in SYSTEM_PROMPT

    def test_prompt_describes_run_states(self):
        assert "STANDBY" in SYSTEM_PROMPT
        assert "PREPARING" in SYSTEM_PROMPT
        assert "RUNNING" in SYSTEM_PROMPT
        assert "SUCCESS" in SYSTEM_PROMPT
        assert "FAILED" in SYSTEM_PROMPT

    def test_prompt_describes_roles(self):
        assert "Coordinator" in SYSTEM_PROMPT
        assert "Participant" in SYSTEM_PROMPT

    def test_prompt_describes_workflow(self):
        assert "Register" in SYSTEM_PROMPT
        assert "Create project" in SYSTEM_PROMPT
        assert "Upload datasets" in SYSTEM_PROMPT
        assert "Download artifacts" in SYSTEM_PROMPT

    def test_prompt_mentions_multi_site(self):
        assert "env_file" in SYSTEM_PROMPT
        assert "multi-site" in SYSTEM_PROMPT.lower()

    def test_prompt_lists_task_types(self):
        assert "LogisticRegression" in SYSTEM_PROMPT
        assert "CoxProportionalHazards" in SYSTEM_PROMPT
        assert "KaplanMeier" in SYSTEM_PROMPT
        assert "PoissonRegression" in SYSTEM_PROMPT

    def test_prompt_describes_task_config(self):
        assert "total_round" in SYSTEM_PROMPT
        assert "current_round" in SYSTEM_PROMPT

    def test_prompt_describes_components(self):
        assert "Router" in SYSTEM_PROMPT
        assert "Controller" in SYSTEM_PROMPT

    def test_prompt_mentions_failure_handling(self):
        assert "failure" in SYSTEM_PROMPT.lower() or "failed" in SYSTEM_PROMPT.lower()
        assert "logs" in SYSTEM_PROMPT.lower()
