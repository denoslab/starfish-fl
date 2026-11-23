"""
How a site polls the router for its runs and hands each status change to its task.

The Celery tasks ``fetch_run``, ``monitor_run`` and ``process_task`` in
``starfish/celery.py`` call these functions with a run cache in Redis.
starfish-site calls the same functions from one process with a cache in
memory. Neither Celery nor Redis is imported here.

A run cache remembers the run record last dispatched per run id. ``get``
returns that record or None, and ``put`` stores one.
"""

import json
import logging
import os
import threading
import time

import requests

from starfish.controller.router_auth import router_auth
from starfish.controller.utils import load_class, camel_to_snake, format_status, epoch_time_in_sec

logger = logging.getLogger(__name__)

# A cached run expires after a day, as the Redis keys always did. A run that
# keeps one status for longer than that is dispatched again.
RUN_TTL_SECONDS = 86400
# Coordinator statuses that are dispatched again, as retries, while they last
RETRY_STATUSES = ('preparing', 'pending_aggregating')
ROUTER_TIMEOUT_SECONDS = 30
# Task instances unused for this long are dropped
INSTANCE_TTL_SECONDS = 86400

# One task instance per run and task sequence, kept between status changes
ml_models = dict()
ml_models_alive = dict()


def _router_url():
    return os.getenv('ROUTER_URL')


class MemoryRunCache:
    """A run cache for a single process, with the same expiry as the Redis one."""

    def __init__(self, ttl=RUN_TTL_SECONDS, clock=time.monotonic):
        self._ttl = ttl
        self._clock = clock
        self._runs = {}
        self._lock = threading.Lock()

    def get(self, run_id):
        with self._lock:
            item = self._runs.get(run_id)
            if item is None:
                return None
            run, stamp = item
            if self._clock() - stamp >= self._ttl:
                del self._runs[run_id]
                return None
            return run

    def put(self, run):
        with self._lock:
            self._runs[run['id']] = (run, self._clock())

    def clear(self):
        with self._lock:
            self._runs.clear()


def fetch():
    """The router's active runs, or None when it answers with an error."""
    response = requests.get('{0}/runs/active/'.format(_router_url()),
                            headers={'Content-type': 'application/json'},
                            auth=router_auth(),
                            data=json.dumps(dict()),
                            timeout=ROUTER_TIMEOUT_SECONDS)
    if response.ok:
        return response.json()
    return None


def changed_runs(run_list, site_uid, cache):
    """
    This site's runs whose status differs from the one last dispatched.

    Each of them is cached as dispatched.
    """
    changed = []
    for r in run_list or []:
        if r['site_uid'] != site_uid:
            continue
        cached = cache.get(r['id'])
        if cached is None or format_status(r['status']) != format_status(cached['status']):
            changed.append(r)
            cache.put(r)
    return changed


def retry_runs(run_list, site_uid, cache):
    """This site's coordinator runs still in a retry status since they were dispatched."""
    retry = []
    for r in run_list or []:
        if r['site_uid'] != site_uid or r['role'] != 'coordinator':
            continue
        cached = cache.get(r['id'])
        if cached is None:
            continue
        status = format_status(r['status'])
        if status == format_status(cached['status']) and status in RETRY_STATUSES:
            retry.append(r)
    return retry


def progress(run):
    """Status, task and round of a run record, for telling stale messages apart."""
    cur_seq = run.get('cur_seq')
    try:
        current_round = run['tasks'][cur_seq -
                                     1]['config'].get('current_round')
    except (KeyError, IndexError, TypeError):
        current_round = None
    return format_status(run.get('status')), cur_seq, current_round


def is_stale(run):
    """
    True when the router no longer has the run in the state this message carries.

    A message can wait in the queue while the processor is down; after a restart
    it would act on an old status or round, for example skip a later round's
    validation. The current state was queued again at startup, so dropping the
    old one loses nothing. If the router cannot be asked, the message is kept.
    """
    try:
        response = requests.get('{0}/runs/{1}/'.format(_router_url(), run['id']),
                                auth=router_auth(), timeout=10)
    except requests.RequestException:
        return False
    if not response.ok:
        return False
    return progress(response.json()) != progress(run)


def refresh_model(model_id):
    """Mark a task instance as used and drop the ones unused for a day."""
    now = epoch_time_in_sec()
    ml_models_alive[model_id] = now
    for old in [k for k, v in ml_models_alive.items() if v < now - INSTANCE_TTL_SECONDS]:
        ml_models.pop(old, None)
        ml_models_alive.pop(old, None)


def process(run, is_retry):
    """
    Call the run's task for the run's current status.

    :param run: run record from the router
    :param is_retry: whether it is a retry of an unchanged coordinator status
    """
    run_id = run['id']
    cur_seq = run['cur_seq']
    model = run['tasks'][cur_seq - 1]['model']
    status = format_status(run['status'])
    model_id = "{}_{}".format(run_id, cur_seq)
    if is_stale(run):
        logger.info("Dropped a stale message for run {}: {} is no longer its state".format(
            run_id, status))
        return

    try:
        klass = None
        module_path = 'starfish.controller.tasks.{}.task'.format(
            camel_to_snake(model))
        try:
            klass = load_class(module_path, model)
        except (ImportError, AttributeError):
            pass

        if klass is None:
            raise ImportError(f"Model {model} not found in any module path")

        if model_id in ml_models:
            instance = ml_models[model_id]
        else:
            instance = klass(run)
            ml_models[model_id] = instance
        refresh_model(model_id)
        instance.method_call(status, run, is_retry)

    except (ImportError, AttributeError) as e:
        logger.warning("{} not found with error: {}".format(model, e))
