import json
import warnings
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy import stats as scipy_stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from starfish.controller.file.file_utils import (
    gen_mid_artifacts_url, gen_all_mid_artifacts_url, gen_artifacts_url,
    downloaded_artifacts_url
)
from starfish.controller.tasks.abstract_task import AbstractTask
from starfish.controller.tasks.diagnostics import compute_vif, residual_summary, shapiro_wilk_test

warnings.filterwarnings('ignore')

MIN_SAMPLE_SIZE = 30
DEFAULT_NUM_IMPUTATIONS = 5
DEFAULT_MAX_ITER = 10


class MultipleImputation(AbstractTask):
    """
    Federated Multiple Imputation using MICE (Multiple Imputation by
    Chained Equations).

    Each site runs MICE locally via sklearn's IterativeImputer to create
    M imputed datasets, fits a linear regression on each, and pools
    results using Rubin's rules.  The coordinator aggregates across sites
    using inverse-variance weighted meta-analysis with Rubin's rules for
    combining within- and between-imputation variance.

    Data format: CSV with features in all columns except the last.
    Last column is the continuous outcome.  Missing values are encoded
    as empty cells (read as NaN by pandas).

    Config options (in task JSON config):
      - m: number of imputations (default 5)
      - max_iter: max MICE iterations (default 10)
    """

    def __init__(self, run):
        super().__init__(run)
        self.sample_size = None
        self.X_raw = None
        self.y_raw = None
        self.m = DEFAULT_NUM_IMPUTATIONS
        self.max_iter = DEFAULT_MAX_ITER

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

        self.X_raw = X
        self.y_raw = y

        # Read config
        config = self.tasks[self.cur_seq - 1].get('config', {})
        self.m = config.get('m', DEFAULT_NUM_IMPUTATIONS)
        self.max_iter = config.get('max_iter', DEFAULT_MAX_ITER)

        self.logger.debug(
            'Data shape: {} features x {} samples, m={}, max_iter={}'.format(
                X.shape[1], n, self.m, self.max_iter))

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
        self.logger.info('Starting Multiple Imputation (MICE) training...')
        try:
            # Combine X and y for joint imputation
            full_data = np.column_stack([self.X_raw, self.y_raw])
            n_cols = full_data.shape[1]
            n_features = n_cols - 1

            # Compute missingness diagnostics
            missing_mask = np.isnan(full_data)
            missingness_fractions = np.nanmean(
                missing_mask, axis=0).tolist()
            complete_cases = int(np.sum(~np.any(missing_mask, axis=1)))

            # Run MICE: create M imputed datasets and fit OLS on each
            coef_list = []
            var_list = []

            for imp_idx in range(self.m):
                imputer = IterativeImputer(
                    max_iter=self.max_iter,
                    random_state=imp_idx,
                    sample_posterior=True,
                )
                imputed = imputer.fit_transform(full_data)

                X_imp = sm.add_constant(imputed[:, :n_features])
                y_imp = imputed[:, n_features]

                model = sm.OLS(y_imp, X_imp).fit()
                coef_list.append(model.params)
                var_list.append(model.bse ** 2)

            # Pool using Rubin's rules (within-site)
            coef_arr = np.array(coef_list)  # (m, p)
            var_arr = np.array(var_list)     # (m, p)

            pooled_coef = np.mean(coef_arr, axis=0)
            within_var = np.mean(var_arr, axis=0)
            between_var = np.var(coef_arr, axis=0, ddof=1)
            total_var = within_var + (1 + 1.0 / self.m) * between_var
            pooled_se = np.sqrt(total_var)

            # Degrees of freedom (Barnard-Rubin adjustment)
            lambda_hat = (
                (1 + 1.0 / self.m) * between_var / (total_var + 1e-10))
            df_old = (self.m - 1) / (lambda_hat ** 2 + 1e-10)
            df_obs = ((self.sample_size - n_features - 1) + 1) / (
                (self.sample_size - n_features - 1) + 3) * (
                self.sample_size - n_features - 1) * (1 - lambda_hat)
            df_adjusted = np.where(
                np.isinf(df_old), df_obs,
                (df_old * df_obs) / (df_old + df_obs))
            df_adjusted = np.maximum(df_adjusted, 1.0)

            t_values = pooled_coef / (pooled_se + 1e-10)
            p_values = 2 * scipy_stats.t.sf(
                np.abs(t_values), df_adjusted)

            # CIs using t-distribution with adjusted df
            t_crit = scipy_stats.t.ppf(0.975, df_adjusted)
            ci_lower = pooled_coef - t_crit * pooled_se
            ci_upper = pooled_coef + t_crit * pooled_se

            n_total_features = n_features + 1  # includes intercept
            feature_names = ['const'] + [
                'x{}'.format(i) for i in range(n_total_features - 1)]

            # Diagnostics from first imputed dataset
            diagnostics = {}
            try:
                first_X = sm.add_constant(
                    IterativeImputer(
                        max_iter=self.max_iter, random_state=0
                    ).fit_transform(full_data)[:, :n_features])
                first_y = full_data[:, n_features]
                first_model = sm.OLS(
                    np.where(np.isnan(first_y), 0, first_y),
                    first_X).fit()
                diagnostics['vif'] = compute_vif(first_X[:, 1:])
                diagnostics['residual_summary'] = residual_summary(
                    first_model.resid)
                diagnostics['shapiro_wilk'] = shapiro_wilk_test(
                    first_model.resid)
            except Exception:
                pass

            stats = {
                'sample_size': self.sample_size,
                'complete_cases': complete_cases,
                'm': self.m,
                'coef': pooled_coef.tolist(),
                'se': pooled_se.tolist(),
                'within_var': within_var.tolist(),
                'between_var': between_var.tolist(),
                't_values': t_values.tolist(),
                'p_values': p_values.tolist(),
                'ci_lower': ci_lower.tolist(),
                'ci_upper': ci_upper.tolist(),
                'df': df_adjusted.tolist(),
                'missingness_fractions': missingness_fractions,
                'feature_names': feature_names,
                'diagnostics': diagnostics,
            }

            url = gen_mid_artifacts_url(
                self.run_id, self.cur_seq, self.get_round())
            self.logger.info("Upload: {} \n to: {}".format(stats, url))
            return self.save_artifacts(url, json.dumps(stats))
        except Exception as e:
            self.logger.error(
                'Error during Multiple Imputation training: {}'.format(e))
            return False

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

        # Inverse-variance weighted meta-analysis with Rubin's rules
        n_coefs = len(mid_artifacts[0]['coef'])
        weighted_coef = np.zeros(n_coefs)
        weight_sum = np.zeros(n_coefs)
        total_samples = 0
        total_complete = 0

        # Collect per-site estimates for between-site variance
        site_coefs = []
        site_ses = []

        for art in mid_artifacts:
            coef = np.array(art['coef'])
            se = np.array(art['se'])
            w = 1.0 / (se ** 2 + 1e-10)
            weighted_coef += w * coef
            weight_sum += w
            total_samples += art['sample_size']
            total_complete += art.get('complete_cases', art['sample_size'])
            site_coefs.append(coef)
            site_ses.append(se)

        pooled_coef = weighted_coef / weight_sum
        pooled_se = np.sqrt(1.0 / weight_sum)

        # Between-site variance (adds to uncertainty)
        site_coefs_arr = np.array(site_coefs)
        k = len(mid_artifacts)
        if k > 1:
            between_site_var = np.var(site_coefs_arr, axis=0, ddof=1)
            total_var = pooled_se ** 2 + (1 + 1.0 / k) * between_site_var
            pooled_se = np.sqrt(total_var)

        t_values = pooled_coef / (pooled_se + 1e-10)
        # Use normal approximation for federated pooling
        p_values = (2 * (1 - scipy_stats.norm.cdf(
            np.abs(t_values)))).tolist()
        ci_lower = (pooled_coef - 1.96 * pooled_se).tolist()
        ci_upper = (pooled_coef + 1.96 * pooled_se).tolist()

        # Aggregate missingness fractions as weighted average
        agg_missing = np.zeros(
            len(mid_artifacts[0].get('missingness_fractions', [])))
        for art in mid_artifacts:
            fracs = np.array(art.get('missingness_fractions', []))
            if len(fracs) > 0:
                agg_missing += fracs * art['sample_size']
        if total_samples > 0 and len(agg_missing) > 0:
            agg_missing = (agg_missing / total_samples).tolist()
        else:
            agg_missing = []

        result = {
            'sample_size': total_samples,
            'complete_cases': total_complete,
            'n_sites': k,
            'coef': pooled_coef.tolist(),
            'se': pooled_se.tolist(),
            't_values': t_values.tolist(),
            'p_values': p_values,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'missingness_fractions': agg_missing,
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
