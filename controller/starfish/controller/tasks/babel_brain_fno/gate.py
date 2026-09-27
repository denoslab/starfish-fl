"""Evaluation gate for BabelBrainFno, SF-05.

The coordinator scores the current global model and each aggregated
candidate on its own held-out store, ``BABELBRAIN_FL_EVAL_STORE``, which
holds the test subjects exported with the same sample contract. The
candidate becomes the new global model only if it does not regress by
more than the configured margins. The defaults allow no regression at all,
confirmed on 2026-09-27, answer T8, knowing that a noisy focal metric on a
small eval set can then hold the current model for several rounds.

Config, under the task's ``gate`` key::

    {"enabled": true, "max_rel_l2_increase": 0.0, "max_peak_distance_increase_mm": 0.0}

Both margins compare means over the eval store. ``rel_l2`` is in percent
points. Disabling the gate is explicit; a missing eval store fails the round.

When the eval store is a schema 1.1 test set with region labels, decision
D6, the report also gives both models' scores per target region. The gate
still decides on the overall means.
"""

import hashlib

import numpy as np

from starfish.controller.tasks.babel_brain_fno import metrics as M
from starfish.controller.tasks.babel_brain_fno.store import REGIONS

EVAL_STORE_ENV = 'BABELBRAIN_FL_EVAL_STORE'
DEFAULT_GATE = {'enabled': True, 'max_rel_l2_increase': 0.0,
                'max_peak_distance_increase_mm': 0.0}
# The per-region breakdown uses the schema 1.1 region labels of the NeuroFUS test set, D6.
# An eval store without labels gives no breakdown.
REGION_BREAKDOWN_AVAILABLE = True


def gate_config(config):
    cfg = dict(DEFAULT_GATE)
    cfg.update(config.get('gate') or {})
    return cfg


def store_digest(records):
    """Identifies the eval set, so a cached score is reused only for the same samples."""
    h = hashlib.sha256()
    for r in sorted(records, key=lambda r: r.sample_id):
        h.update('{}:{}\n'.format(r.sample_id, r.sha256).encode())
    return h.hexdigest()


def evaluate(arrays, complex_keys, pkg, bucket_hz, records, device):
    """Score model weights on eval records.

    Returns ``(summary, regions)``: the six metrics summarised over all
    samples, and per region class, or None when no record has a region.
    """
    import torch
    from torch.utils.data import DataLoader

    from starfish.controller.tasks.babel_brain_fno import weights as W
    from starfish.controller.tasks.babel_brain_fno.sample_io import torch_dataset

    model = pkg.build_model(bucket_hz)
    model.load_state_dict(W.arrays_to_state(arrays, complex_keys))
    model.to(device).eval()
    rows = []
    with torch.no_grad():
        for batch in DataLoader(torch_dataset(records), batch_size=1):
            x, y, aux = pkg.inputs(batch)
            pred = model(x.to(device)).float().cpu().numpy()
            rows.append(M.sample_metrics(pred[0], y.numpy()[0],
                                         aux['brain_mask'].numpy()[0],
                                         float(np.asarray(batch['spacing_mm']).reshape(-1)[0])))
    return M.summarize(rows), by_region(rows, records)


def by_region(rows, records):
    """Summaries per region class, in the contract's order, or None without labels."""
    labelled = [(r.region, row) for r, row in zip(records, rows) if r.region]
    if not labelled:
        return None
    return {region: M.summarize([row for label, row in labelled if label == region])
            for region in REGIONS if any(label == region for label, _ in labelled)}


def region_breakdown(current, candidate):
    """Per region, the current and the candidate model's summaries, or None."""
    if not current and not candidate:
        return None
    current, candidate = current or {}, candidate or {}
    regions = list(candidate) + [r for r in current if r not in candidate]
    return {r: {'current': current.get(r), 'candidate': candidate.get(r)}
            for r in regions}


def decide(candidate, current, cfg):
    """``(accepted, reasons)``: reject when a margin is exceeded or a score is not finite."""
    reasons = []
    checks = (('rel_l2_pct', cfg['max_rel_l2_increase']),
              ('peak_distance_mm', cfg['max_peak_distance_increase_mm']))
    for name, margin in checks:
        new, old = candidate[name]['mean'], current[name]['mean']
        if new is None or not np.isfinite(new):
            reasons.append('{} of the candidate is not finite'.format(name))
        elif old is not None and new > old + float(margin):
            reasons.append('{} rose from {:.4f} to {:.4f}, more than the margin {}'.format(
                name, old, new, margin))
    return not reasons, reasons
