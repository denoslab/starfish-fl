import json
import os
from celery import Celery
from celery.signals import beat_init, worker_init
from celery.utils.log import get_task_logger
from kombu import Exchange, Queue
from starfish.controller import redis, run_dispatch
from starfish.controller.site_status_task import report_alive

logger = get_task_logger(__name__)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "starfish.settings")
app = Celery("starfish")
app.config_from_object("django.conf:settings", namespace="CELERY")

run_exchange = Exchange('starfish.run', type='direct')
task_exchange = Exchange('starfish.processor', type='direct')

app.conf.task_queues = {
    Queue('starfish.run', run_exchange, routing_key='starfish.run'),
    Queue('starfish.processor', task_exchange,
          routing_key='starfish.processor')
}

router_url = os.getenv('ROUTER_URL')
router_username = os.getenv('ROUTER_USERNAME')
router_password = os.getenv('ROUTER_PASSWORD')
site_id = os.getenv('SITE_UID')

run_key = 'starfish:controller:run:list:'

# The polling and dispatch logic lives in run_dispatch, shared with starfish-site
is_stale = run_dispatch.is_stale
_progress = run_dispatch.progress
ml_models = run_dispatch.ml_models
ml_models_alive = run_dispatch.ml_models_alive
refresh_model = run_dispatch.refresh_model


class RedisRunCache:
    """The run cache of run_dispatch, kept in Redis for all worker processes."""

    def get(self, run_id):
        value = redis.get_redis().get(run_key + str(run_id))
        return json.loads(value) if value else None

    def put(self, run):
        redis.get_redis().set(run_key + str(run['id']), json.dumps(run),
                              ex=run_dispatch.RUN_TTL_SECONDS)


run_cache = RedisRunCache()


@app.task(bind=True, queue='starfish.run', name='fetch_run')
def fetch_run(args):
    """
    Fetch run status based on site_id, cache it and notify processor.
    :return:
    """
    logger.debug("Received: {}".format(args))

    runs = check_status_change(site_id)
    if runs:
        for run in runs:
            process_task.s(run, False).apply_async(
                queue='starfish.processor')


@app.task(bind=True, queue='starfish.run', name='monitor_run')
def monitor_run(args):
    """
    Monitor coordinator run in waiting status
    :param args:
    :return:
    """
    for run in run_dispatch.retry_runs(fetch(), site_id, run_cache):
        process_task.s(run, True).apply_async(
            queue='starfish.processor')


@app.task(bind=True, queue='starfish.run', name='site_heartbeat')
def heartbeat(args):
    report_alive()


@app.task(bind=True, queue='starfish.processor', name='process_task')
def process_task(args, run, is_retry):
    """
    :param args:
    :param run: run model
    :param is_retry: whether the task is triggered by retry
    :return:
    """
    logger.debug("Received: {} options: {}".format(args, run))
    run_dispatch.process(run, is_retry)


def fetch():
    return run_dispatch.fetch()


def check_status_change(site_uid) -> []:
    """
    Check run status to decide whether send message to task processor
    :return:
    """
    return run_dispatch.changed_runs(fetch(), site_uid, run_cache) or None


def get_run_from_redis(run):
    r = redis.get_redis()
    return r.get(run_key + str(run['id']))


def add_to_redis(run):
    run_cache.put(run)


def remove_from_redis(run_id):
    r = redis.get_redis()
    k = run_key + str(run_id)
    r.delete(k)


def reset_cache():
    r = redis.get_redis()
    for key in r.scan_iter(run_key+'*'):
        r.delete(key)


@worker_init.connect
@beat_init.connect
def reset_cache_on_start(**kwargs):
    """
    Empty the run cache when a worker or beat starts, so every active run is
    dispatched again. Importing this module does not touch Redis.
    """
    reset_cache()


app.autodiscover_tasks()
