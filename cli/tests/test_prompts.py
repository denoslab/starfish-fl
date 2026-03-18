"""
Tests for the system prompt content.
"""

from starfish_cli.agent.prompts import SYSTEM_PROMPT, EXPERIMENT_SYSTEM_PROMPT


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


class TestExperimentSystemPrompt:
    """Verify the experiment prompt extends the base prompt with experiment guidance."""

    def test_experiment_prompt_extends_base(self):
        assert SYSTEM_PROMPT in EXPERIMENT_SYSTEM_PROMPT

    def test_experiment_prompt_is_longer(self):
        assert len(EXPERIMENT_SYSTEM_PROMPT) > len(SYSTEM_PROMPT)

    def test_experiment_prompt_has_decision_tree(self):
        assert "Decision Tree" in EXPERIMENT_SYSTEM_PROMPT

    def test_experiment_prompt_has_heuristics(self):
        assert "Heuristics" in EXPERIMENT_SYSTEM_PROMPT

    def test_experiment_prompt_mentions_analyze_dataset(self):
        assert "analyze_dataset" in EXPERIMENT_SYSTEM_PROMPT

    def test_experiment_prompt_mentions_interpret_results(self):
        assert "interpret_results" in EXPERIMENT_SYSTEM_PROMPT

    def test_experiment_prompt_mentions_compare_experiments(self):
        assert "compare_experiments" in EXPERIMENT_SYSTEM_PROMPT

    def test_experiment_prompt_has_refinement_guidance(self):
        assert "Iterative Refinement" in EXPERIMENT_SYSTEM_PROMPT

    def test_experiment_prompt_has_report_format(self):
        assert "Report Format" in EXPERIMENT_SYSTEM_PROMPT

    def test_experiment_prompt_mentions_round_guidance(self):
        assert "Single-round models" in EXPERIMENT_SYSTEM_PROMPT
        assert "Iterative models" in EXPERIMENT_SYSTEM_PROMPT

    def test_experiment_prompt_has_interpretation_thresholds(self):
        assert "AUC > 0.8" in EXPERIMENT_SYSTEM_PROMPT
        assert "VIF > 10" in EXPERIMENT_SYSTEM_PROMPT
        assert "Concordance > 0.7" in EXPERIMENT_SYSTEM_PROMPT

    def test_experiment_prompt_mentions_recommend_task(self):
        assert "recommend_task" in EXPERIMENT_SYSTEM_PROMPT

    def test_experiment_prompt_mentions_generate_config(self):
        assert "generate_config" in EXPERIMENT_SYSTEM_PROMPT
