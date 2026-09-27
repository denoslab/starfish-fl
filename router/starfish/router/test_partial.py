"""Tests for partial participation on the router, SF-10."""

from starfish.router.models import Run
from starfish.router.test_transfer import TransferTestCase, fresh


class SitOutTest(TransferTestCase):

    def status(self, run, state, **extra):
        body = {'status': state, 'update_all': True}
        body.update(extra)
        return self.client.put('/starfish/api/v1/runs/{}/status/'.format(run.id), body,
                               format='json')

    def statuses(self):
        return [fresh(r).status for r in self.runs]

    def test_sit_out_marks_those_runs_and_moves_the_rest(self):
        before = fresh(self.co).updated_at
        self.assertEqual(self.status(self.co, Run.RunStatus.RUNNING, sit_out=[
                         self.pa2.id]).status_code, 202)
        self.assertEqual(self.statuses(), [Run.RunStatus.RUNNING, Run.RunStatus.RUNNING,
                                           Run.RunStatus.SITTING_OUT])
        self.assertGreater(fresh(self.co).updated_at, before)

    def test_later_steps_leave_sitting_out_runs_alone(self):
        self.status(self.co, Run.RunStatus.RUNNING, sit_out=[self.pa2.id])
        self.status(self.co, Run.RunStatus.AGGREGATING)
        self.assertEqual(self.statuses(), [Run.RunStatus.AGGREGATING, Run.RunStatus.AGGREGATING,
                                           Run.RunStatus.SITTING_OUT])

    def test_next_round_brings_every_site_back(self):
        self.status(self.co, Run.RunStatus.RUNNING, sit_out=[self.pa2.id])
        self.status(self.co, Run.RunStatus.STANDBY, increase_round=True)
        self.assertEqual(self.statuses(), [Run.RunStatus.STANDBY] * 3)
        self.assertEqual(fresh(self.pa2).tasks[0]['config'].get('current_round'),
                         fresh(self.co).tasks[0]['config'].get('current_round'))

    def test_the_coordinator_never_sits_out(self):
        self.status(self.co, Run.RunStatus.RUNNING, sit_out=[self.co.id])
        self.assertEqual(fresh(self.co).status, Run.RunStatus.RUNNING)

    def test_without_sit_out_every_run_moves_as_before(self):
        self.status(self.co, Run.RunStatus.RUNNING)
        self.assertEqual(self.statuses(), [Run.RunStatus.RUNNING] * 3)
