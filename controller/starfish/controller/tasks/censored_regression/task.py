import json
import warnings
from pathlib import Path

import numpy as np
from scipy import optimize, stats

from starfish.controller.file.file_utils import (
    gen_mid_artifacts_url, gen_all_mid_artifacts_url, gen_artifacts_url,
    downloaded_artifacts_url
)
from starfish.controller.tasks.abstract_task import AbstractTask
from starfish.controller.tasks.diagnostics import (
    compute_vif, residual_summary, shapiro_wilk_test
)

warnings.filterwarnings('ignore')


def _tobit_neg_log_likelihood(params, X, y, censor):
    """Negative log-likelihood for Tobit Type I model.

    Parameters
    ----------
    params : array  [beta_0, beta_1, ..., beta_p, log_sigma]
    X : ndarray (n, p+1)  design matrix WITH intercept
    y : ndarray (n,)  outcome
    censor : ndarray (n,)  0=observed, 1=right-censored, -1=left-censored
    """
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)

    mu = X @ beta
    z = (y - mu) / sigma

    ll = 0.0
    # Observed
    obs = censor == 0
    if np.any(obs):
        ll += np.sum(-log_sigma - 0.5 * np.log(2 * np.pi) - 0.5 * z[obs] ** 2)

    # Right-censored
    rc = censor == 1
    if np.any(rc):
        log_sf = stats.norm.logsf(z[rc])
        ll += np.sum(log_sf)

    # Left-censored
    lc = censor == -1
    if np.any(lc):
        log_cdf = stats.norm.logcdf(z[lc])
        ll += np.sum(log_cdf)

    return -ll


def _tobit_neg_log_likelihood_gradient(params, X, y, censor):
    """Gradient of the negative log-likelihood for Tobit Type I model."""
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)

    mu = X @ beta
    z = (y - mu) / sigma

    grad_beta = np.zeros_like(beta)
    grad_log_sigma = 0.0

    # Observed
    obs = censor == 0
    if np.any(obs):
        grad_beta += np.sum(X[obs] * (z[obs] / sigma)[:, None], axis=0)
        grad_log_sigma += np.sum(z[obs] ** 2 - 1)

    # Right-censored: d/dz log(1 - Phi(z)) = -phi(z) / (1 - Phi(z))
    # d/dbeta = (d/dz) * (dz/dbeta) = (d/dz) * (-X/sigma)
    rc = censor == 1
    if np.any(rc):
        dldz = -stats.norm.pdf(z[rc]) / (stats.norm.sf(z[rc]) + 1e-300)
        grad_beta -= np.sum(X[rc] * (dldz / sigma)[:, None], axis=0)
        grad_log_sigma -= np.sum(dldz * z[rc])

    # Left-censored: d/dz log(Phi(z)) = phi(z) / Phi(z)
    # d/dbeta = (d/dz) * (dz/dbeta) = (d/dz) * (-X/sigma)
    lc = censor == -1
    if np.any(lc):
        dldz = stats.norm.pdf(z[lc]) / (stats.norm.cdf(z[lc]) + 1e-300)
        grad_beta -= np.sum(X[lc] * (dldz / sigma)[:, None], axis=0)
        grad_log_sigma -= np.sum(dldz * z[lc])

    return -np.append(grad_beta, grad_log_sigma)


class CensoredRegression(AbstractTask):
    """
    Federated censored regression (Tobit Type I) via MLE.

    Data format: CSV with features in all columns except the last two.
    Second-to-last column is the outcome (continuous, possibly censored).
    Last column is the censoring indicator:
      0 = observed, 1 = right-censored, -1 = left-censored.
    """

    def __init__(self, run):
        super().__init__(run)
        self.sample_size = None
        self.X_train = None
        self.y_train = None
        self.censor_train = None
        self.X_test = None
        self.y_test = None
        self.censor_test = None
        self.feature_names = None

    def prepare_data(self) -> bool:
        self.logger.debug(
            'Loading dataset for run {} ...'.format(self.run_id))
        X, y = self.read_dataset(self.run_id)
        if X is None or len(X) == 0 or y is None or len(y) == 0:
            self.logger.warning("Dataset is not ready")
            return False

        # Last column of X is outcome, y is censoring indicator
        features = X[:, :-1]
        outcome = X[:, -1]
        censor = y

        # Validate censoring indicator
        valid_values = {-1, 0, 1}
        if not set(np.unique(censor).astype(int)).issubset(valid_values):
            self.logger.warning(
                "Invalid censoring values. Expected {-1, 0, 1}")
            return False

        n = len(outcome)
        self.sample_size = n

        n_features = features.shape[1]
        self.feature_names = [f'x{i}' for i in range(n_features)]

        # Train/test split (80/20)
        np.random.seed(42)
        idx = np.random.permutation(n)
        split = int(0.8 * n)

        # Add intercept
        X_design = np.column_stack([np.ones(n), features])

        self.X_train = X_design[idx[:split]]
        self.y_train = outcome[idx[:split]]
        self.censor_train = censor[idx[:split]]
        self.X_test = X_design[idx[split:]]
        self.y_test = outcome[idx[split:]]
        self.censor_test = censor[idx[split:]]

        self.logger.debug(
            f'Training data shape: {self.X_train.shape}')
        self.logger.debug(
            f'Test data shape: {self.X_test.shape}')

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
        self.logger.info('Starting Tobit regression training...')

        n_params = self.X_train.shape[1]
        # Initial values: OLS on observed data + log(std(y))
        obs_mask = self.censor_train == 0
        if np.sum(obs_mask) > n_params:
            beta_init, _, _, _ = np.linalg.lstsq(
                self.X_train[obs_mask], self.y_train[obs_mask], rcond=None)
            resid = self.y_train[obs_mask] - self.X_train[obs_mask] @ beta_init
            log_sigma_init = np.log(np.std(resid) + 1e-6)
        else:
            beta_init = np.zeros(n_params)
            log_sigma_init = np.log(np.std(self.y_train) + 1e-6)

        x0 = np.append(beta_init, log_sigma_init)

        result = optimize.minimize(
            _tobit_neg_log_likelihood,
            x0,
            args=(self.X_train, self.y_train, self.censor_train),
            jac=_tobit_neg_log_likelihood_gradient,
            method='L-BFGS-B',
            options={'maxiter': 1000},
        )

        if not result.success:
            self.logger.warning(
                "Optimization did not converge: {}".format(result.message))

        beta_hat = result.x[:-1]
        log_sigma_hat = result.x[-1]
        sigma_hat = np.exp(log_sigma_hat)

        # Standard errors from inverse Hessian
        try:
            hess_inv = result.hess_inv
            if hasattr(hess_inv, 'todense'):
                hess_inv = hess_inv.todense()
            se_all = np.sqrt(np.diag(np.atleast_2d(hess_inv)))
            se_beta = se_all[:-1]
        except Exception:
            se_beta = np.full(len(beta_hat), np.nan)

        # Statistics
        z_values = beta_hat / (se_beta + 1e-10)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_values)))
        ci_lower = beta_hat - 1.96 * se_beta
        ci_upper = beta_hat + 1.96 * se_beta

        # Diagnostics
        diagnostics = self._compute_diagnostics(
            beta_hat, sigma_hat, result.fun)

        to_upload = {
            'sample_size': self.sample_size,
            'coef': beta_hat.tolist(),
            'se': se_beta.tolist(),
            'sigma': float(sigma_hat),
            'p_values': p_values.tolist(),
            'ci_lower': ci_lower.tolist(),
            'ci_upper': ci_upper.tolist(),
            'log_likelihood': float(-result.fun),
            'feature_names': ['intercept'] + self.feature_names,
            'diagnostics': diagnostics,
        }

        url = gen_mid_artifacts_url(
            self.run_id, self.cur_seq, self.get_round())
        self.logger.info("Upload: {} \n to: {}".format(to_upload, url))
        return self.save_artifacts(url, json.dumps(to_upload))

    def _compute_diagnostics(self, beta, sigma, neg_ll):
        """Compute Tobit-specific diagnostics."""
        diag = {}

        # VIF on features (exclude intercept)
        X_no_const = self.X_train[:, 1:]
        if X_no_const.shape[1] > 0:
            diag['vif'] = compute_vif(X_no_const)

        # Generalized residuals for observed data
        obs_mask = self.censor_train == 0
        if np.any(obs_mask):
            mu = self.X_train[obs_mask] @ beta
            resid = self.y_train[obs_mask] - mu
            diag['residual_summary'] = residual_summary(resid)
            diag['shapiro_wilk'] = shapiro_wilk_test(resid)

        # Censoring summary
        n = len(self.censor_train)
        diag['censoring_summary'] = {
            'n_observed': int(np.sum(self.censor_train == 0)),
            'n_right_censored': int(np.sum(self.censor_train == 1)),
            'n_left_censored': int(np.sum(self.censor_train == -1)),
            'pct_censored': float(np.mean(self.censor_train != 0) * 100),
        }

        # Information criteria
        n_params = len(beta) + 1  # beta + sigma
        n_obs = len(self.censor_train)
        ll = -neg_ll
        diag['aic'] = float(-2 * ll + 2 * n_params)
        diag['bic'] = float(-2 * ll + n_params * np.log(n_obs))

        return diag

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
        sigma_weighted = 0.0
        sigma_weight_sum = 0.0

        for art in mid_artifacts:
            coef = np.array(art['coef'])
            se = np.array(art['se'])
            w = 1.0 / (se ** 2 + 1e-10)
            weighted_coef += w * coef
            weight_sum += w
            total_samples += art['sample_size']
            # Pool sigma by sample-size weighting
            sigma_weighted += art['sigma'] * art['sample_size']
            sigma_weight_sum += art['sample_size']

        pooled_coef = weighted_coef / weight_sum
        pooled_se = np.sqrt(1.0 / weight_sum)
        pooled_sigma = sigma_weighted / sigma_weight_sum
        z_values = pooled_coef / pooled_se
        p_values = (2 * (1 - stats.norm.cdf(
            np.abs(z_values)))).tolist()
        ci_lower = (pooled_coef - 1.96 * pooled_se).tolist()
        ci_upper = (pooled_coef + 1.96 * pooled_se).tolist()

        result = {
            'sample_size': total_samples,
            'coef': pooled_coef.tolist(),
            'se': pooled_se.tolist(),
            'sigma': float(pooled_sigma),
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
                    self.logger.debug(
                        "Loaded previous model: {}".format(model))
