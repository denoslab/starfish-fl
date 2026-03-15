import json
import warnings
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy import stats as scipy_stats

from starfish.controller.file.file_utils import (
    gen_mid_artifacts_url, gen_all_mid_artifacts_url, gen_artifacts_url,
    downloaded_artifacts_url
)
from starfish.controller.tasks.abstract_task import AbstractTask
from starfish.controller.tasks.diagnostics import glm_diagnostics

warnings.filterwarnings('ignore')

MIN_SAMPLE_SIZE = 30


class PoissonRegression(AbstractTask):
    """
    Federated Poisson Regression using statsmodels GLM.

    Data format: CSV with features in all columns except the last two.
    Second-to-last column is the offset (log-exposure), last column is
    the count outcome (non-negative integer).
    """

    def __init__(self, run):
        super().__init__(run)
        self.sample_size = None
        self.X = None
        self.y = None
        self.offset = None
        self.model_result = None

    def prepare_data(self) -> bool:
        self.logger.debug(
            'Loading dataset for run {} ...'.format(self.run_id))
        X, y = self.read_dataset(self.run_id)
        if X is None or len(X) == 0 or y is None or len(y) == 0:
            self.logger.warning("Dataset is not ready")
            return False

        n = len(y)
        self.sample_size = n

        if self.sample_size < MIN_SAMPLE_SIZE:
            self.logger.warning(
                "Sample size ({}) is below minimum threshold ({}).".format(
                    self.sample_size, MIN_SAMPLE_SIZE))

        # Validate count data
        if np.any(y < 0):
            self.logger.error("Count variable contains negative values")
            return False

        # Last column of X is offset (log-exposure), rest are features
        self.offset = X[:, -1]
        features = X[:, :-1]
        self.X = sm.add_constant(features)
        self.y = y

        self.logger.debug(
            'Training data shape: {}'.format(self.X.shape))

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
        self.logger.info('Starting Poisson Regression training...')
        try:
            model = sm.GLM(
                self.y, self.X,
                family=sm.families.Poisson(),
                offset=self.offset
            )
            self.model_result = model.fit()
            self.logger.info('Poisson Regression training complete.')

            stats = self._calculate_statistics()
            url = gen_mid_artifacts_url(
                self.run_id, self.cur_seq, self.get_round())
            self.logger.info("Upload: {} \n to: {}".format(stats, url))
            return self.save_artifacts(url, json.dumps(stats))
        except Exception as e:
            self.logger.error(
                'Error during Poisson Regression training: {}'.format(e))
            return False

    def _calculate_statistics(self):
        result = self.model_result
        coef = result.params.tolist()
        se = result.bse.tolist()
        z_values = result.tvalues.tolist()
        p_values = result.pvalues.tolist()
        conf_int = result.conf_int().tolist()
        rate_ratios = np.exp(coef).tolist()

        deviance = float(result.deviance)
        pearson_chi2 = float(result.pearson_chi2)
        aic = float(result.aic)

        n_features = self.X.shape[1]
        feature_names = ['const'] + [
            'x{}'.format(i) for i in range(n_features - 1)]

        # Diagnostics
        diagnostics = glm_diagnostics(self.X, self.y, result)

        return {
            'sample_size': self.sample_size,
            'coef': coef,
            'se': se,
            'z_values': z_values,
            'p_values': p_values,
            'ci_lower': [ci[0] for ci in conf_int],
            'ci_upper': [ci[1] for ci in conf_int],
            'rate_ratios': rate_ratios,
            'deviance': deviance,
            'pearson_chi2': pearson_chi2,
            'aic': aic,
            'feature_names': feature_names,
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

        if not mid_artifacts:
            self.logger.warning(
                "No mid-artifacts found for aggregation")
            return False

        # Inverse-variance weighted meta-analysis of log-rate ratios
        n_coefs = len(mid_artifacts[0]['coef'])
        weighted_coef = np.zeros(n_coefs)
        weight_sum = np.zeros(n_coefs)
        total_samples = 0

        for art in mid_artifacts:
            coef = np.array(art['coef'])
            se = np.array(art['se'])
            w = 1.0 / (se ** 2 + 1e-10)
            weighted_coef += w * coef
            weight_sum += w
            total_samples += art['sample_size']

        pooled_coef = weighted_coef / weight_sum
        pooled_se = np.sqrt(1.0 / weight_sum)
        pooled_rr = np.exp(pooled_coef).tolist()
        z_values = pooled_coef / pooled_se
        p_values = (2 * (1 - scipy_stats.norm.cdf(
            np.abs(z_values)))).tolist()
        ci_lower = (pooled_coef - 1.96 * pooled_se).tolist()
        ci_upper = (pooled_coef + 1.96 * pooled_se).tolist()

        # Pool deviance and AIC as weighted averages
        total_deviance = sum(a['deviance'] for a in mid_artifacts)
        total_pearson_chi2 = sum(a['pearson_chi2'] for a in mid_artifacts)

        result = {
            'sample_size': total_samples,
            'coef': pooled_coef.tolist(),
            'se': pooled_se.tolist(),
            'z_values': z_values.tolist(),
            'p_values': p_values,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'rate_ratios': pooled_rr,
            'deviance': total_deviance,
            'pearson_chi2': total_pearson_chi2,
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
                    self.logger.debug(
                        "Loaded previous model: {}".format(model))
