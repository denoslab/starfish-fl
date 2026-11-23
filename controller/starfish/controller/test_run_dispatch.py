"""Polling and dispatch shared by the Celery workers and starfish-site."""

from unittest import TestCase
from unittest.mock import MagicMock, patch

from starfish.controller import run_dispatch

SITE = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
OTHER = 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'


def run(run_id=1, status='Standby', site=SITE, role='coordinator', model='X'):
    return {'id': run_id, 'status': status, 'site_uid': site, 'role': role, 'cur_seq': 1,
            'tasks': [{'model': model, 'config': {'current_round': 1}}]}


class Clock:

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class MemoryRunCacheTest(TestCase):

    def test_a_cached_run_expires_after_its_ttl(self):
        clock = Clock()
        cache = run_dispatch.MemoryRunCache(ttl=10, clock=clock)
        cache.put(run())
        clock.now = 9.9
        self.assertEqual(cache.get(1)['status'], 'Standby')
        clock.now = 10.0
        self.assertIsNone(cache.get(1))

    def test_clear_forgets_every_run(self):
        cache = run_dispatch.MemoryRunCache()
        cache.put(run(1))
        cache.put(run(2))
        cache.clear()
        self.assertIsNone(cache.get(1))
        self.assertIsNone(cache.get(2))


class ChangedRunsTest(TestCase):

    def test_new_and_changed_runs_of_this_site_are_dispatched_once(self):
        cache = run_dispatch.MemoryRunCache()
        first = [run(1), run(2, site=OTHER)]
        self.assertEqual([r['id'] for r in run_dispatch.changed_runs(first, SITE, cache)], [1])
        self.assertEqual(run_dispatch.changed_runs(first, SITE, cache), [])
        moved = [run(1, status='Preparing')]
        self.assertEqual([r['status'] for r in run_dispatch.changed_runs(moved, SITE, cache)],
                         ['Preparing'])
        self.assertEqual(cache.get(1)['status'], 'Preparing')

    def test_no_answer_from_the_router_dispatches_nothing(self):
        self.assertEqual(run_dispatch.changed_runs(None, SITE, run_dispatch.MemoryRunCache()), [])


class RetryRunsTest(TestCase):

    def test_only_waiting_coordinator_runs_are_retried(self):
        cache = run_dispatch.MemoryRunCache()
        runs = [run(1, 'Preparing'), run(2, 'Pending Aggregating'), run(3, 'Running'),
                run(4, 'Preparing', role='participant'), run(5, 'Preparing', site=OTHER)]
        for r in runs:
            cache.put(r)
        self.assertEqual([r['id'] for r in run_dispatch.retry_runs(runs, SITE, cache)], [1, 2])

    def test_a_run_is_retried_only_after_its_change_was_dispatched(self):
        cache = run_dispatch.MemoryRunCache()
        cache.put(run(1, 'Standby'))
        self.assertEqual(run_dispatch.retry_runs([run(1, 'Preparing')], SITE, cache), [])
        self.assertEqual(run_dispatch.retry_runs([run(7, 'Preparing')], SITE, cache), [])


class ProcessTest(TestCase):

    def setUp(self):
        run_dispatch.ml_models.clear()
        run_dispatch.ml_models_alive.clear()

    def tearDown(self):
        run_dispatch.ml_models.clear()
        run_dispatch.ml_models_alive.clear()

    def test_one_task_instance_serves_every_status_of_a_run(self):
        klass = MagicMock()
        with patch.object(run_dispatch, 'is_stale', return_value=False), \
                patch.object(run_dispatch, 'load_class', return_value=klass):
            run_dispatch.process(run(status='Standby'), False)
            run_dispatch.process(run(status='Preparing'), True)
        klass.assert_called_once()
        calls = klass.return_value.method_call.call_args_list
        self.assertEqual([c.args[0] for c in calls], ['standby', 'preparing'])
        self.assertEqual([c.args[2] for c in calls], [False, True])

    def test_instances_unused_for_a_day_are_dropped(self):
        run_dispatch.ml_models.update({'old_1': object(), 'older_1': object(), 'new_1': object()})
        run_dispatch.ml_models_alive.update({'old_1': 0, 'older_1': 0, 'new_1': 10 ** 12})
        with patch.object(run_dispatch, 'epoch_time_in_sec', return_value=10 ** 12):
            run_dispatch.refresh_model('new_1')
        self.assertEqual(set(run_dispatch.ml_models), {'new_1'})
        self.assertEqual(set(run_dispatch.ml_models_alive), {'new_1'})

    def test_an_unknown_model_is_logged_and_skipped(self):
        with patch.object(run_dispatch, 'is_stale', return_value=False), \
                self.assertLogs(run_dispatch.logger, 'WARNING'):
            run_dispatch.process(run(model='NoSuchModel'), False)
        self.assertEqual(run_dispatch.ml_models, {})


class FetchTest(TestCase):

    def test_fetch_asks_for_active_runs_with_a_timeout(self):
        response = MagicMock(ok=True)
        response.json.return_value = [run()]
        with patch.dict('os.environ', {'ROUTER_URL': 'https://router.example/api'}), \
                patch.object(run_dispatch.requests, 'get', return_value=response) as get:
            self.assertEqual(run_dispatch.fetch(), [run()])
        self.assertEqual(get.call_args.args[0], 'https://router.example/api/runs/active/')
        self.assertEqual(get.call_args.kwargs['timeout'], run_dispatch.ROUTER_TIMEOUT_SECONDS)

    def test_an_error_answer_gives_none(self):
        with patch.object(run_dispatch.requests, 'get', return_value=MagicMock(ok=False)):
            self.assertIsNone(run_dispatch.fetch())
