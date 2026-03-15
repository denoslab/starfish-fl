import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from starfish.controller.file.file_utils import (
    gen_mid_artifacts_url, gen_all_mid_artifacts_url, gen_artifacts_url,
    downloaded_artifacts_url
)
from starfish.controller.tasks.abstract_task import AbstractTask

warnings.filterwarnings('ignore')


class KaplanMeier(AbstractTask):
    """
    Federated Kaplan-Meier survival estimation with log-rank test.

    Data format: CSV with group column (1st), feature columns (middle),
    time column (2nd-to-last), event column (last, 1=event, 0=censored).

    For KM estimation, only group, time, and event columns are used.
    """

    def __init__(self, run):
        super().__init__(run)
        self.sample_size = None
        self.group = None
        self.time = None
        self.event = None

    def prepare_data(self) -> bool:
        self.logger.debug(
            'Loading dataset for run {} ...'.format(self.run_id))
        X, y = self.read_dataset(self.run_id)
        if X is None or len(X) == 0 or y is None or len(y) == 0:
            self.logger.warning("Dataset is not ready")
            return False

        # First column is group, second-to-last is time, last (y) is event
        self.group = X[:, 0]
        self.time = X[:, -1]
        self.event = y
        self.sample_size = len(y)

        self.logger.debug(
            f'Sample size: {self.sample_size}, '
            f'Events: {self.event.sum():.0f}, '
            f'Groups: {np.unique(self.group).tolist()}')
        return True

    def validate(self) -> bool:
        task_round = self.get_round()
        self.logger.debug(
            "Run {} - task {} - round {} task begins".format(
                self.run_id, self.cur_seq, task_round))
        return self.download_artifact()

    def training(self) -> bool:
        self.logger.info('Starting Kaplan-Meier estimation...')
        results = self._compute_km()
        url = gen_mid_artifacts_url(
            self.run_id, self.cur_seq, self.get_round())
        self.logger.info("Upload KM results to: {}".format(url))
        return self.save_artifacts(url, json.dumps(results))

    def _compute_km(self):
        groups = np.unique(self.group)
        km_results = {}

        for g in groups:
            mask = self.group == g
            kmf = KaplanMeierFitter()
            kmf.fit(self.time[mask], self.event[mask], label=f'group_{int(g)}')

            survival_table = kmf.survival_function_
            timeline = survival_table.index.tolist()
            survival_prob = survival_table.iloc[:, 0].tolist()

            median_survival = float(kmf.median_survival_time_)
            ci = kmf.confidence_interval_survival_function_
            ci_lower = ci.iloc[:, 0].tolist()
            ci_upper = ci.iloc[:, 1].tolist()

            km_results[f'group_{int(g)}'] = {
                'timeline': timeline,
                'survival_probability': survival_prob,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'median_survival': median_survival
                    if not np.isinf(median_survival) else None,
                'n_observations': int(mask.sum()),
                'n_events': int(self.event[mask].sum()),
            }

        # Log-rank test (if exactly 2 groups)
        logrank_result = None
        if len(groups) == 2:
            mask0 = self.group == groups[0]
            mask1 = self.group == groups[1]
            lr = logrank_test(
                self.time[mask0], self.time[mask1],
                self.event[mask0], self.event[mask1])
            logrank_result = {
                'test_statistic': float(lr.test_statistic),
                'p_value': float(lr.p_value),
            }

        # Build at-risk table for federated pooling
        at_risk_table = self._build_at_risk_table()

        return {
            'sample_size': self.sample_size,
            'km_results': km_results,
            'logrank': logrank_result,
            'at_risk_table': at_risk_table,
        }

    def _build_at_risk_table(self):
        """Build at-risk table for federated pooling of KM estimates."""
        groups = np.unique(self.group)
        table = {}
        for g in groups:
            mask = self.group == g
            t = self.time[mask]
            e = self.event[mask]
            unique_times = np.sort(np.unique(t[e == 1]))
            events = []
            at_risk = []
            for ut in unique_times:
                n_at_risk = int(np.sum(t >= ut))
                n_events = int(np.sum((t == ut) & (e == 1)))
                events.append(n_events)
                at_risk.append(n_at_risk)
            table[f'group_{int(g)}'] = {
                'times': unique_times.tolist(),
                'events': events,
                'at_risk': at_risk,
            }
        return table

    def do_aggregate(self) -> bool:
        mid_artifacts = []
        directory = gen_all_mid_artifacts_url(self.project_id, self.batch_id)
        for path in Path(directory).rglob(
                "*-{}-{}-mid-artifacts".format(self.cur_seq, self.get_round())):
            with open(str(path), 'r') as f:
                for line in f:
                    mid_artifacts.append(json.loads(line))

        if not mid_artifacts:
            self.logger.warning("No mid-artifacts found for aggregation")
            return False

        # Pool at-risk tables across sites
        pooled = self._pool_km(mid_artifacts)

        url = gen_artifacts_url(
            self.run_id, self.cur_seq, self.get_round())
        self.logger.info("Upload pooled KM to: {}".format(url))
        if self.save_artifacts(url, json.dumps(pooled)):
            self.upload(True)
            return True
        return False

    def _pool_km(self, mid_artifacts):
        """Pool KM estimates from multiple sites by combining at-risk tables."""
        total_samples = sum(a['sample_size'] for a in mid_artifacts)

        # Collect all group names
        all_groups = set()
        for art in mid_artifacts:
            all_groups.update(art['at_risk_table'].keys())

        pooled_km = {}
        for group in sorted(all_groups):
            # Merge event times across sites
            all_times = set()
            for art in mid_artifacts:
                if group in art['at_risk_table']:
                    all_times.update(art['at_risk_table'][group]['times'])
            all_times = sorted(all_times)

            # Pool events and at-risk counts
            pooled_events = []
            pooled_at_risk = []
            for t in all_times:
                total_events = 0
                total_at_risk = 0
                for art in mid_artifacts:
                    if group not in art['at_risk_table']:
                        continue
                    tbl = art['at_risk_table'][group]
                    if t in tbl['times']:
                        idx = tbl['times'].index(t)
                        total_events += tbl['events'][idx]
                        total_at_risk += tbl['at_risk'][idx]
                pooled_events.append(total_events)
                pooled_at_risk.append(total_at_risk)

            # Recompute KM curve from pooled table
            survival = []
            s = 1.0
            for d, n in zip(pooled_events, pooled_at_risk):
                if n > 0:
                    s *= (1.0 - d / n)
                survival.append(s)

            # Compute median survival
            median = None
            for i, s_val in enumerate(survival):
                if s_val <= 0.5:
                    median = all_times[i]
                    break

            total_obs = sum(
                art['at_risk_table'].get(group, {}).get(
                    'at_risk', [0])[0] if art['at_risk_table'].get(
                    group, {}).get('at_risk', []) else 0
                for art in mid_artifacts)
            total_events = sum(
                sum(art['at_risk_table'].get(group, {}).get('events', []))
                for art in mid_artifacts)

            pooled_km[group] = {
                'timeline': all_times,
                'survival_probability': survival,
                'median_survival': median,
                'n_observations': total_obs,
                'n_events': total_events,
            }

        return {
            'sample_size': total_samples,
            'km_results': pooled_km,
        }
