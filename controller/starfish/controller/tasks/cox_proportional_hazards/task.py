import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from starfish.controller.file.file_utils import (
    gen_mid_artifacts_url, gen_all_mid_artifacts_url, gen_artifacts_url,
    downloaded_artifacts_url
)
from starfish.controller.tasks.abstract_task import AbstractTask
from starfish.controller.tasks.diagnostics import cox_diagnostics

warnings.filterwarnings('ignore')


class CoxProportionalHazards(AbstractTask):
    """
    Federated Cox Proportional Hazards regression using lifelines.

    Data format: CSV with features in all columns except the last two.
    Second-to-last column is the time variable, last column is the event
    indicator (1 = event, 0 = censored).
    """

    def __init__(self, run):
        super().__init__(run)
        self.sample_size = None
        self.train_df = None
        self.test_df = None
        self.cph = None

    def prepare_data(self) -> bool:
        self.logger.debug(
            'Loading dataset for run {} ...'.format(self.run_id))
        X, y = self.read_dataset(self.run_id)
        if X is None or len(X) == 0 or y is None or len(y) == 0:
            self.logger.warning("Dataset is not ready")
            return False

        # Last column of X is time, y is event indicator
        features = X[:, :-1]
        time = X[:, -1]
        event = y

        n = len(event)
        self.sample_size = n

        # Build DataFrame with column names
        n_features = features.shape[1]
        col_names = [f'x{i}' for i in range(n_features)]
        df = pd.DataFrame(features, columns=col_names)
        df['time'] = time
        df['event'] = event

        # Train/test split (80/20)
        np.random.seed(42)
        idx = np.random.permutation(n)
        split = int(0.8 * n)
        self.train_df = df.iloc[idx[:split]].copy()
        self.test_df = df.iloc[idx[split:]].copy()

        self.logger.debug(
            f'Training data shape: {self.train_df.shape}')
        self.logger.debug(
            f'Test data shape: {self.test_df.shape}')

        self.cph = CoxPHFitter()

        if not self.is_first_round():
            self._load_previous_model()

        return True

    def validate(self) -> bool:
        task_round = self.get_round()
        self.logger.debug(
            "Run {} - task {} - round {} task begins".format(
                self.run_id, self.cur_seq, task_round))
        return self.download_artifact()

    def training(self) -> bool:
        self.logger.info('Starting Cox PH training...')
        self.cph.fit(
            self.train_df,
            duration_col='time',
            event_col='event'
        )
        self.logger.info('Cox PH training complete.')
        to_upload = self._calculate_statistics()
        url = gen_mid_artifacts_url(
            self.run_id, self.cur_seq, self.get_round())
        self.logger.info("Upload: {} \n to: {}".format(to_upload, url))
        return self.save_artifacts(url, json.dumps(to_upload))

    def _calculate_statistics(self):
        summary = self.cph.summary
        coef = self.cph.params_.values.tolist()
        se = summary['se(coef)'].values.tolist()
        hr = summary['exp(coef)'].values.tolist()
        p_values = summary['p'].values.tolist()
        ci_lower = summary['coef lower 95%'].values.tolist()
        ci_upper = summary['coef upper 95%'].values.tolist()
        concordance = self.cph.concordance_index_

        # Diagnostics
        diagnostics = cox_diagnostics(self.cph, self.train_df)

        return {
            'sample_size': self.sample_size,
            'coef': coef,
            'se': se,
            'hazard_ratio': hr,
            'p_values': p_values,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'concordance_index': concordance,
            'feature_names': list(self.cph.params_.index),
            'diagnostics': diagnostics,
        }

    def do_aggregate(self) -> bool:
        mid_artifacts = []
        directory = gen_all_mid_artifacts_url(self.project_id, self.batch_id)
        for path in Path(directory).rglob(
                "*-{}-{}-mid-artifacts".format(self.cur_seq, self.get_round())):
            with open(str(path), 'r') as f:
                for line in f:
                    mid_artifacts.append(json.loads(line))

        self.logger.debug(
            "Downloaded mid artifacts: {}".format(mid_artifacts))

        if not mid_artifacts:
            self.logger.warning(
                "No mid-artifacts found for aggregation")
            return False

        # Inverse-variance weighted meta-analysis of coefficients
        n_coefs = len(mid_artifacts[0]['coef'])
        weighted_coef = np.zeros(n_coefs)
        weight_sum = np.zeros(n_coefs)
        total_samples = 0

        for art in mid_artifacts:
            coef = np.array(art['coef'])
            se = np.array(art['se'])
            # Inverse variance weights
            w = 1.0 / (np.array(se) ** 2 + 1e-10)
            weighted_coef += w * coef
            weight_sum += w
            total_samples += art['sample_size']

        pooled_coef = weighted_coef / weight_sum
        pooled_se = np.sqrt(1.0 / weight_sum)
        pooled_hr = np.exp(pooled_coef).tolist()
        z_values = pooled_coef / pooled_se
        p_values = (2 * (1 - __import__('scipy').stats.norm.cdf(
            np.abs(z_values)))).tolist()
        ci_lower = (pooled_coef - 1.96 * pooled_se).tolist()
        ci_upper = (pooled_coef + 1.96 * pooled_se).tolist()

        result = {
            'sample_size': total_samples,
            'coef': pooled_coef.tolist(),
            'se': pooled_se.tolist(),
            'hazard_ratio': pooled_hr,
            'p_values': p_values,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'feature_names': mid_artifacts[0].get('feature_names', []),
        }

        url = gen_artifacts_url(
            self.run_id, self.cur_seq, self.get_round())
        self.logger.info("Upload: {} \n to: {}".format(result, url))
        if self.save_artifacts(url, json.dumps(result)):
            self.upload(True)
            return True
        return False

    def _load_previous_model(self):
        seq_no, round_no = self.get_previous_seq_and_round()
        if not seq_no or not round_no:
            return
        directory = downloaded_artifacts_url(self.run_id, seq_no, round_no)
        if not directory:
            return
        for path in Path(directory).rglob(
                "*-{}-{}-artifacts".format(seq_no, round_no)):
            with open(str(path), 'r') as f:
                for line in f:
                    model = json.loads(line)
                    # lifelines doesn't support warm start directly,
                    # but we load previous coefficients for reference
                    self.logger.debug(
                        "Loaded previous model: {}".format(model))
