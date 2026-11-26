"""
One process that does the work of the controller's Celery beat and two workers.

Three timers keep the beat schedule of ``starfish/settings.py``. Every 5 s
``fetch_run`` queues this site's runs whose status changed, every 10 s
``monitor_run`` queues coordinator runs that still wait, and every 5 s the
heartbeat tells the router the site is up. A single processor thread takes
the queue in order, like the ``starfish.processor`` worker with concurrency
1. The polling and the task calls themselves come from
``starfish.controller.run_dispatch``, the code the Celery tasks call, with
the run cache in memory in place of Redis.
"""

import logging
import threading
import time
from collections import deque

logger = logging.getLogger('starfish_site')

POLL_SECONDS = 5
MONITOR_SECONDS = 10
HEARTBEAT_SECONDS = 5
# A failing job logs a warning at first and then once per this many failures
WARN_EVERY = 60


def round_of(run):
    """``, round N of M`` for a run whose task names its rounds, else an empty string."""
    try:
        config = run['tasks'][run['cur_seq'] - 1]['config']
        current, total = config.get('current_round'), config.get('total_round')
    except (KeyError, IndexError, TypeError, AttributeError):
        return ''
    return ', round {} of {}'.format(current, total) if current and total else ''


class SiteRunner:
    """
    Poll, queue and process this site's runs until :meth:`stop`.

    :param site_uid: this site's ``SITE_UID``
    :param dispatch: ``starfish.controller.run_dispatch`` or a stand-in for tests
    :param heartbeat: function that reports this site as alive
    """

    def __init__(self, site_uid, dispatch, heartbeat, cache=None, poll_seconds=POLL_SECONDS,
                 monitor_seconds=MONITOR_SECONDS, heartbeat_seconds=HEARTBEAT_SECONDS,
                 clock=time.monotonic):
        self.site_uid = site_uid
        self.dispatch = dispatch
        self.heartbeat = heartbeat
        self.cache = cache if cache is not None else dispatch.MemoryRunCache()
        self.schedule = (('heartbeat', heartbeat_seconds, self.heartbeat_once),
                         ('fetch_run', poll_seconds, self.poll_once),
                         ('monitor_run', monitor_seconds, self.monitor_once))
        self.clock = clock
        self.stopping = threading.Event()
        self.current = None
        self._queue = deque()
        self._queued = set()
        self._cond = threading.Condition()
        self._failures = {}
        self._threads = []

    # The queue

    def _key(self, run, is_retry):
        status, cur_seq, current_round = self.dispatch.progress(run)
        return run['id'], status, cur_seq, current_round, bool(is_retry)

    def submit(self, run, is_retry):
        """
        Queue a run for the processor. Return False for a repeat that was dropped.

        Each message waits in the queue at most once, and a repeat of a waiting
        message is dropped. A repeat of the message in progress still queues
        behind it, as with Celery.
        """
        key = self._key(run, is_retry)
        with self._cond:
            if key in self._queued:
                return False
            self._queued.add(key)
            self._queue.append((key, run, is_retry))
            self._cond.notify()
        return True

    def pending(self):
        with self._cond:
            return len(self._queue)

    def _take(self):
        with self._cond:
            while not self._queue and not self.stopping.is_set():
                self._cond.wait(1.0)
            if self.stopping.is_set():
                return None
            key, run, is_retry = self._queue.popleft()
            self._queued.discard(key)
            return run, is_retry

    def process_loop(self):
        """Take queued messages in order and hand each to its task, until stopped."""
        while True:
            item = self._take()
            if item is None:
                return
            run, is_retry = item
            self.current = (run['id'], run.get('status'), is_retry)
            started = self.clock()
            logger.info('Run %s: %s%s%s', run['id'], run.get('status'), round_of(run),
                        ', again while it waits' if is_retry else '')
            try:
                self.dispatch.process(run, is_retry)
            except Exception:
                logger.exception('Run %s failed in %s', run['id'], run.get('status'))
            finally:
                self.current = None
            logger.debug('Run %s: %s took %.1f s', run['id'], run.get('status'),
                         self.clock() - started)

    # The timers

    def poll_once(self):
        for run in self.dispatch.changed_runs(self.dispatch.fetch(), self.site_uid, self.cache):
            self.submit(run, False)

    def monitor_once(self):
        for run in self.dispatch.retry_runs(self.dispatch.fetch(), self.site_uid, self.cache):
            self.submit(run, True)

    def heartbeat_once(self):
        self.heartbeat()

    def _succeeded(self, name):
        if self._failures.pop(name, 0) >= 1:
            logger.info('%s works again', name)

    def _failed(self, name, error):
        count = self._failures.get(name, 0) + 1
        self._failures[name] = count
        if count == 1 or count % WARN_EVERY == 0:
            logger.warning('%s failed%s: %s: %s', name,
                           '' if count == 1 else ' {} times in a row'.format(count),
                           error.__class__.__name__, error)
        else:
            logger.debug('%s failed: %s', name, error)

    def every(self, name, seconds, job):
        """Run ``job`` every ``seconds`` until stopped, skipping beats it fell behind on."""
        next_at = self.clock()
        while not self.stopping.is_set():
            try:
                job()
                self._succeeded(name)
            except Exception as e:
                self._failed(name, e)
            next_at += seconds
            delay = next_at - self.clock()
            if delay < 0:
                next_at, delay = self.clock(), 0
            if self.stopping.wait(delay):
                return

    # Lifecycle

    def start(self):
        self._threads = [threading.Thread(target=self.process_loop, name='processor', daemon=True)]
        self._threads += [threading.Thread(target=self.every, args=(name, seconds, job),
                                           name=name, daemon=True)
                          for name, seconds, job in self.schedule]
        for thread in self._threads:
            thread.start()

    def stop(self):
        """Stop the timers and let the processor finish the message in progress."""
        self.stopping.set()
        with self._cond:
            self._cond.notify_all()

    def alive(self):
        return any(t.is_alive() for t in self._threads)

    def join(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        for thread in self._threads:
            left = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(left)
        return not self.alive()
