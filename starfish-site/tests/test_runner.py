"""The single-process runner that stands in for Celery beat, the workers and Redis."""

import threading
import time
import unittest
from types import SimpleNamespace

from starfish_site.runner import SiteRunner

SITE = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'


def run(run_id=1, status='Standby', current_round=1):
    return {'id': run_id, 'status': status, 'site_uid': SITE, 'role': 'coordinator',
            'cur_seq': 1, 'tasks': [{'model': 'X', 'config': {'current_round': current_round}}]}


class FakeCache:

    def __init__(self):
        self.runs = {}

    def get(self, run_id):
        return self.runs.get(run_id)

    def put(self, r):
        self.runs[r['id']] = r


class FakeDispatch:
    """Stands in for starfish.controller.run_dispatch."""

    MemoryRunCache = FakeCache

    def __init__(self, active=None, retry=None, process=None):
        self.active = active or []
        self.retry = retry or []
        self.calls = []
        self.fetches = 0
        self._process = process

    def progress(self, r):
        return r['status'].lower(), r['cur_seq'], r['tasks'][0]['config']['current_round']

    def fetch(self):
        self.fetches += 1
        return self.active

    def changed_runs(self, run_list, site_uid, cache):
        changed = [r for r in run_list if (cache.get(r['id']) or {}).get('status') != r['status']]
        for r in changed:
            cache.put(r)
        return changed

    def retry_runs(self, run_list, site_uid, cache):
        return list(self.retry)

    def process(self, r, is_retry):
        self.calls.append((r['id'], r['status'], is_retry))
        if self._process:
            self._process(r, is_retry)


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class QueueTest(unittest.TestCase):

    def test_a_waiting_repeat_is_dropped(self):
        runner = SiteRunner(SITE, FakeDispatch(), heartbeat=lambda: None)
        self.assertTrue(runner.submit(run(), False))
        self.assertFalse(runner.submit(run(), False))
        self.assertTrue(runner.submit(run(), True))
        self.assertTrue(runner.submit(run(current_round=2), False))
        self.assertEqual(runner.pending(), 3)

    def test_a_repeat_of_the_message_in_progress_queues_again(self):
        runner = SiteRunner(SITE, FakeDispatch(), heartbeat=lambda: None)
        runner.submit(run(), True)
        self.assertEqual(runner._take(), (run(), True))
        self.assertTrue(runner.submit(run(), True))

    def test_poll_and_monitor_queue_changes_and_retries(self):
        dispatch = FakeDispatch(active=[run(1), run(2, 'Preparing')], retry=[run(2, 'Preparing')])
        runner = SiteRunner(SITE, dispatch, heartbeat=lambda: None)
        runner.poll_once()
        runner.poll_once()
        runner.monitor_once()
        queued = [(r['id'], is_retry) for _, r, is_retry in runner._queue]
        self.assertEqual(queued, [(1, False), (2, False), (2, True)])


class ProcessorTest(unittest.TestCase):

    def test_messages_run_in_order_and_an_error_does_not_stop_the_processor(self):
        def explode(r, is_retry):
            if r['id'] == 1:
                raise RuntimeError('task failed')

        dispatch = FakeDispatch(process=explode)
        runner = SiteRunner(SITE, dispatch, heartbeat=lambda: None)
        runner.submit(run(1), False)
        runner.submit(run(2), False)
        thread = threading.Thread(target=runner.process_loop)
        with self.assertLogs('starfish_site', 'ERROR'):
            thread.start()
            self.assertTrue(wait_until(lambda: len(dispatch.calls) == 2))
        runner.stop()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual([c[0] for c in dispatch.calls], [1, 2])

    def test_stop_finishes_the_step_in_progress_and_leaves_the_rest(self):
        release = threading.Event()
        dispatch = FakeDispatch(process=lambda r, is_retry: release.wait(5))
        runner = SiteRunner(SITE, dispatch, heartbeat=lambda: None)
        runner.submit(run(1), False)
        runner.submit(run(2), False)
        thread = threading.Thread(target=runner.process_loop)
        thread.start()
        self.assertTrue(wait_until(lambda: runner.current is not None))
        runner.stop()
        release.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual([c[0] for c in dispatch.calls], [1])
        self.assertEqual(runner.pending(), 1)


class TimerTest(unittest.TestCase):

    def test_a_failing_job_warns_once_and_says_when_it_recovers(self):
        outcomes = iter([OSError('router down'), OSError('router down'), None, None])

        def job():
            error = next(outcomes, None)
            if error:
                raise error

        runner = SiteRunner(SITE, FakeDispatch(), heartbeat=lambda: None)
        with self.assertLogs('starfish_site', 'DEBUG') as logs:
            thread = threading.Thread(target=runner.every, args=('fetch_run', 0.01, job))
            thread.start()
            time.sleep(0.2)
            runner.stop()
            thread.join(2)
        warnings = [r for r in logs.records if r.levelname == 'WARNING']
        self.assertEqual(len(warnings), 1)
        self.assertIn('router down', warnings[0].getMessage())
        self.assertTrue(any('works again' in r.getMessage() for r in logs.records))


class LifecycleTest(unittest.TestCase):

    def test_the_runner_polls_processes_beats_and_stops(self):
        beats = []
        dispatch = FakeDispatch(active=[run(1, 'Standby')])
        runner = SiteRunner(SITE, dispatch, heartbeat=lambda: beats.append(1),
                            poll_seconds=0.02, monitor_seconds=0.05, heartbeat_seconds=0.02)
        runner.start()
        self.assertTrue(wait_until(lambda: dispatch.calls and len(beats) >= 2))
        dispatch.active = [run(1, 'Preparing')]
        self.assertTrue(wait_until(lambda: len(dispatch.calls) == 2))
        runner.stop()
        self.assertTrue(runner.join(timeout=3))
        self.assertEqual(dispatch.calls, [(1, 'Standby', False), (1, 'Preparing', False)])


if __name__ == '__main__':
    unittest.main()
