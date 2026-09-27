"""No agent code in the BabelBrain path, SF-06.

The probe runs in a fresh interpreter, because this test process has
already imported the agent modules through other tests.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from starfish.controller.tasks.abstract_task import AbstractTask

CONTROLLER_DIR = Path(__file__).resolve().parents[5]


def run_probe(*args):
    # Clear the site switch so the probe tests the task, not this environment
    env = dict(os.environ, STARFISH_DISABLE_AGENTS='')
    result = subprocess.run(
        [sys.executable, '-m', 'starfish.controller.tasks.babel_brain_fno.probe', *args],
        cwd=CONTROLLER_DIR, capture_output=True, text=True, timeout=120, env=env)
    lines = result.stdout.strip().splitlines()
    report = json.loads(lines[-1]) if lines else None
    return result.returncode, report, result.stderr


class ProbeTest(TestCase):

    def test_babelbrain_round_loads_no_agent_module(self):
        code, report, stderr = run_probe()
        self.assertIsNotNone(report, stderr)
        self.assertEqual(report['agent_modules_loaded'], [], stderr)
        self.assertEqual(code, 0, stderr)
        # Standby to Preparing without upload, then Running, then training fails
        self.assertEqual(report['statuses'][:2], [3, 4])

    def test_probe_detects_agent_imports(self):
        """Negative control: the probe is not vacuous."""
        code, report, stderr = run_probe('--allow-agents')
        self.assertIsNotNone(report, stderr)
        self.assertEqual(code, 1)
        self.assertIn('starfish.controller.agent.hooks',
                      report['agent_modules_loaded'])


class _Task(AbstractTask):
    def validate(self):
        return True

    def prepare_data(self):
        return True

    def training(self):
        return True

    def do_aggregate(self):
        return True


RUN = {'id': 9, 'project': 1, 'batch': 1, 'role': 'participant',
       'status': 'Standby', 'cur_seq': 1,
       'tasks': [{'seq': 1, 'model': 'X', 'config': {
           'total_round': 1, 'current_round': 1, 'agent': {'enabled': True}}}]}


@patch('starfish.controller.file.file_utils.base_folder',
       os.path.join(tempfile.gettempdir(), 'starfish-agents-off-test'))
class SiteSwitchTest(TestCase):

    def test_agents_load_by_default(self):
        with patch.dict(os.environ, {'STARFISH_DISABLE_AGENTS': ''}):
            task = _Task(RUN)
        self.assertTrue(task._agent_hooks.enabled)

    def test_site_switch_turns_agents_off_for_any_task(self):
        for value in ('1', 'true', 'YES'):
            with patch.dict(os.environ, {'STARFISH_DISABLE_AGENTS': value}):
                task = _Task(RUN)
            self.assertIsNone(task._agent_hooks, value)
