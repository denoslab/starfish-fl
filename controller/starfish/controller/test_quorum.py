"""Tests for partial participation in AbstractTask, SF-10."""

import os
import tempfile
from unittest import TestCase
from unittest.mock import patch

from starfish.controller.tasks.abstract_task import AbstractTask


class _Task(AbstractTask):
    def validate(self):
        return True

    def prepare_data(self):
        return True

    def training(self):
        return True

    def do_aggregate(self):
        return True


def run_dict(role='coordinator', **config):
    cfg = {'total_round': 3, 'current_round': 1}
    cfg.update(config)
    return {'id': 1, 'project': 1, 'batch': 1, 'role': role, 'status': 'Preparing',
            'cur_seq': 1, 'tasks': [{'seq': 1, 'model': 'X', 'config': cfg}]}


def runs(*states):
    """Router run records: the first is the coordinator."""
    return [{'id': i + 1, 'role': 'coordinator' if i == 0 else 'participant', 'status': st,
             'updated_at': '2026-09-27T06:00:00Z'} for i, st in enumerate(states)]


@patch('starfish.controller.file.file_utils.base_folder',
       os.path.join(tempfile.gettempdir(), 'starfish-quorum-test'))
class RoundQuorumTest(TestCase):

    def task(self, **config):
        return _Task(run_dict(**config))

    def quorum(self, task, records, expected='preparing', elapsed=0.0):
        with patch.object(_Task, 'fetch_runs', return_value=records), \
                patch.object(_Task, '_seconds_since', return_value=elapsed):
            return task.round_quorum(expected)

    def test_without_min_participants_every_site_must_be_ready(self):
        task = self.task()
        self.assertEqual(self.quorum(task, runs(
            'Preparing', 'Preparing')), ('proceed', []))
        self.assertEqual(self.quorum(task, runs(
            'Preparing', 'Standby')), ('wait', []))
        self.assertEqual(self.quorum(task, runs(
            'Preparing', 'Failed')), ('fail', []))

    def test_quorum_goes_on_without_a_failed_site(self):
        task = self.task(min_participants=2)
        action, sit_out = self.quorum(
            task, runs('Preparing', 'Preparing', 'Failed'))
        self.assertEqual((action, sit_out), ('proceed', [3]))

    def test_waits_for_a_late_site_until_the_deadline(self):
        task = self.task(min_participants=2, round_deadline_minutes=1)
        records = runs('Preparing', 'Preparing', 'Standby')
        self.assertEqual(self.quorum(task, records, elapsed=30), ('wait', []))
        self.assertEqual(self.quorum(
            task, records, elapsed=61), ('proceed', [3]))

    def test_without_a_deadline_it_waits_for_every_active_site(self):
        task = self.task(min_participants=2)
        self.assertEqual(self.quorum(task, runs('Preparing', 'Preparing', 'Standby'),
                                     elapsed=10 ** 6), ('wait', []))

    def test_deadline_needs_the_quorum_to_be_ready(self):
        task = self.task(min_participants=3, round_deadline_minutes=1)
        self.assertEqual(self.quorum(task, runs('Preparing', 'Preparing', 'Standby'),
                                     elapsed=600), ('wait', []))

    def test_fails_when_too_few_sites_remain_or_the_coordinator_fails(self):
        task = self.task(min_participants=2)
        self.assertEqual(self.quorum(task, runs(
            'Preparing', 'Failed', 'Pending Failed'))[0], 'fail')
        self.assertEqual(self.quorum(task, runs(
            'Failed', 'Preparing', 'Preparing'))[0], 'fail')

    def test_sitting_out_sites_do_not_count(self):
        task = self.task(min_participants=2)
        records = runs('Pending Aggregating',
                       'Pending Aggregating', 'Sitting Out')
        self.assertEqual(self.quorum(
            task, records, 'pending_aggregating'), ('proceed', []))

    @patch.object(_Task, 'notify')
    def test_coordinator_step_sends_the_sites_to_sit_out(self, notify):
        task = self.task(min_participants=2, round_deadline_minutes=1)
        records = runs('Preparing', 'Preparing', 'Standby')
        with patch.object(_Task, 'fetch_runs', return_value=records), \
                patch.object(_Task, '_seconds_since', return_value=120):
            task.preparing(run_dict(min_participants=2,
                           round_deadline_minutes=1))
        notify.assert_called_once_with(
            4, param={'update_all': True, 'sit_out': [3]})

    @patch.object(_Task, 'notify')
    def test_a_site_sitting_out_does_nothing(self, notify):
        task = _Task(run_dict(role='participant'))
        task.sitting_out(run_dict(role='participant'))
        notify.assert_not_called()
        self.assertEqual(task.status, 'sitting_out')

    def test_seconds_since_parses_router_timestamps(self):
        self.assertGreater(AbstractTask._seconds_since(
            '2020-01-01T00:00:00Z'), 1e8)
        self.assertGreater(AbstractTask._seconds_since(
            '2020-01-01T00:00:00.123456+00:00'), 1e8)
        self.assertEqual(AbstractTask._seconds_since(None), 0.0)
