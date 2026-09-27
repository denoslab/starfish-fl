"""Check that the BabelBrain path never loads agent code.

Runs one BabelBrainFno round, standby to pending_failed, against a synthetic
store in a temporary folder, with the router calls stubbed out. Then lists
any agent or LLM module that got imported. Run it in a fresh interpreter::

    python -m starfish.controller.tasks.babel_brain_fno.probe [--require-no-anthropic] [--site-store]

Exit code 0 when clean, 1 otherwise. Prints a JSON report. Importing the
``starfish`` package connects to Redis, so Redis must be reachable.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from unittest import mock

AGENT_MODULES = ('starfish.controller.agent', 'anthropic')


def loaded_agent_modules():
    return sorted(name for name in sys.modules
                  if any(name == p or name.startswith(p + '.') for p in AGENT_MODULES))


def site_store_report():
    """Counts and an ID digest for this site's own store, and whether it is read-only."""
    from starfish.controller.tasks.babel_brain_fno.store import SampleStore
    store = SampleStore.from_env()
    ids = sorted(r.sample_id for r in store.samples())
    try:
        probe_file = os.path.join(store.root, '.write-probe')
        with open(probe_file, 'w'):
            pass
        os.remove(probe_file)
        read_only = False
    except OSError:
        read_only = True
    return {
        'counts': store.counts(250000),
        'rejections': dict(store.rejections),
        'ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest()[:16],
        'read_only': read_only,
    }


def run_round(allow_agents=False):
    """Drive one round through the lifecycle; return the notified statuses."""
    from starfish.controller.file import file_utils
    from starfish.controller.tasks.babel_brain_fno.store import STORE_ENV
    from starfish.controller.tasks.babel_brain_fno.synthetic import write_synthetic_store
    from starfish.controller.tasks.babel_brain_fno.task import BabelBrainFno

    with tempfile.TemporaryDirectory() as tmp:
        store = os.path.join(tmp, 'store')
        write_synthetic_store(store, n_groups=6, per_group=4, val_groups=1)
        config = {
            'total_round': 1, 'current_round': 1,
            'data_source': {'type': 'babelbrain_store', 'bucket_hz': 250000},
            # Ask for the agent: the task must refuse it anyway.
            'agent': {'enabled': True, 'summaries': True},
        }
        run = {'id': 1, 'project': 1, 'batch': 1, 'role': 'coordinator',
               'status': 'Standby', 'cur_seq': 1,
               'tasks': [{'seq': 1, 'model': 'BabelBrainFno', 'config': config}]}
        with mock.patch.object(file_utils, 'base_folder', os.path.join(tmp, 'local')), \
                mock.patch.dict(os.environ, {STORE_ENV: store}), \
                mock.patch.object(BabelBrainFno, 'agents_allowed', allow_agents), \
                mock.patch.object(BabelBrainFno, 'notify') as notify, \
                mock.patch.object(BabelBrainFno, 'upload', return_value=True), \
                mock.patch.object(BabelBrainFno, 'runs_in_fails', return_value=False), \
                mock.patch.object(BabelBrainFno, 'runs_in_same_state', return_value=True):
            task = BabelBrainFno(run)
            task.standby(run)
            task.preparing(run)
            task.running(run)
            task.pending_failed(run)
            return [c.args[0] for c in notify.call_args_list]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--require-no-anthropic', action='store_true',
                        help='also fail if the anthropic package is installed')
    parser.add_argument('--allow-agents', action='store_true',
                        help='negative control: let the task load agent hooks')
    parser.add_argument('--site-store', action='store_true',
                        help='also check the store named by BABELBRAIN_FL_STORE')
    args = parser.parse_args(argv)

    statuses = run_round(allow_agents=args.allow_agents)
    report = {
        'statuses': statuses,
        'agent_modules_loaded': loaded_agent_modules(),
        'anthropic_installed': importlib.util.find_spec('anthropic') is not None,
    }
    clean = not report['agent_modules_loaded'] and \
        not (args.require_no_anthropic and report['anthropic_installed'])
    if args.site_store:
        try:
            report['site_store'] = site_store_report()
            clean = clean and report['site_store']['read_only'] and \
                report['site_store']['counts']['train'] > 0
        except Exception as e:
            report['site_store'] = {'error': e.__class__.__name__}
            clean = False
    report['clean'] = clean
    print(json.dumps(report))
    return 0 if clean else 1


if __name__ == '__main__':
    sys.exit(main())
