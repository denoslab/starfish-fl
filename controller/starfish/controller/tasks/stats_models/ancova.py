"""
ANCOVA (Analysis of Covariance) for Federated Learning

ANCOVA combines ANOVA and regression - it tests for differences between group means
while controlling for continuous covariates.

Statistical outputs:
- Coefficients with standard errors, p-values, confidence intervals
- F-statistics for group effects
- Partial eta-squared (effect size)
- Adjusted group means

Federated approach:
- Each site computes local OLS (Ordinary Least Squares) regression with group dummies + covariates
- Sites share: coefficients, standard errors, sample size, SS components
- Coordinator aggregates via inverse-variance weighted meta-analysis
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
import statsmodels.api as sm

from starfish.controller.file.file_utils import (
    gen_mid_artifacts_url, gen_all_mid_artifacts_url, gen_artifacts_url,
    downloaded_artifacts_url
)
from starfish.controller.tasks.abstract_task import AbstractTask
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

MIN_SAMPLE_SIZE = 30


class Ancova(AbstractTask):
    """
    ANCOVA implementation for federated statistical analysis.
    
    Expected data format:
    - Last column: continuous outcome variable (Y)
    - First K columns: group indicator(s) - should be dummy/one-hot encoded
    - Remaining columns: continuous covariates
    
    Task config should specify:
    - n_group_columns: number of columns representing group membership (dummy coded)
    - total_round: number of FL rounds (typically 1 for ANCOVA)
    - current_round: current round number
    """

    def __init__(self, run):
        super().__init__(run)
        self.sample_size = None
        self.X = None
        self.y = None
        self.X_with_const = None
        self.model_result = None
        self.n_group_cols = 1  # default, can be overridden by config

    def prepare_data(self) -> bool:
        self.logger.debug('Loading dataset for run {} ...'.format(self.run_id))
        X, y = self.read_dataset(self.run_id)
        
        if X is None or len(X) == 0 or y is None or len(y) == 0:
            self.logger.warning("Dataset is not ready")
            return False
        
        self.sample_size = len(y)
        
        # Privacy check
        if self.sample_size < MIN_SAMPLE_SIZE:
            self.logger.warning(
                f"Sample size ({self.sample_size}) is below minimum threshold ({MIN_SAMPLE_SIZE}). "
                "This may pose privacy risks."
            )
        
        # Get config for number of group columns
        task_config = self.tasks[self.cur_seq - 1].get('config', {})
        self.n_group_cols = task_config.get('n_group_columns', 1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Add constant for statsmodels OLS
        self.X_with_const = sm.add_constant(self.X)
        
        self.logger.debug(f'Training data shape: {self.X.shape}')
        self.logger.debug(f'Training label shape: {self.y.shape}')
        self.logger.debug(f'Number of group columns: {self.n_group_cols}')
        
        # Load previous round artifacts if not first round
        if not self.is_first_round():
            seq_no, round_no = self.get_previous_seq_and_round()
            directory = downloaded_artifacts_url(self.run_id, seq_no, round_no)
            for path in Path(directory).rglob("*-{}-{}-artifacts".format(seq_no, round_no)):
                with open(str(path), 'r') as f:
                    for line in f:
                        prev_model = json.loads(line)
                        self.logger.debug(f"Loaded previous artifacts: {prev_model.keys()}")
        
        return True

    def validate(self) -> bool:
        task_round = self.get_round()
        self.logger.debug(
            "Run {} - task {} - round {} task begins".format(
                self.run_id, self.cur_seq, task_round
            )
        )
        return self.download_artifact()

    def training(self) -> bool:
        """
        Fit OLS regression model and compute ANCOVA statistics.
        """
        self.logger.info('Starting ANCOVA analysis...')
        
        try:
            # Fit OLS model
            model = sm.OLS(self.y, self.X_with_const)
            self.model_result = model.fit()
            
            self.logger.info(f'Model fitted. R² = {self.model_result.rsquared:.4f}')
            
            # Calculate and save statistics
            stats = self.calculate_statistics()
            
            url = gen_mid_artifacts_url(self.run_id, self.cur_seq, self.get_round())
            self.logger.info(f"Saving mid-artifacts to: {url}")
            
            return self.save_artifacts(url, json.dumps(stats))
            
        except Exception as e:
            self.logger.error(f'Error during ANCOVA training: {e}')
            return False

    def calculate_statistics(self) -> dict:
        """
        Calculate ANCOVA statistics for federated aggregation.
        """
        result = self.model_result
        
        # Basic coefficients and inference
        coef = result.params.tolist() # coefficients: the estimated effects (weights) for each variable in the model
        std_err = result.bse.tolist() # the standard errors for each coefficient
        t_values = result.tvalues.tolist() # t-statistics for each coefficient
        p_values = result.pvalues.tolist() # p-values for each coefficient
        conf_int = result.conf_int().tolist() # confidence intervals for each coefficient
        
        # Model fit statistics
        r_squared = result.rsquared  # the proportion of variance explained by the model
        adj_r_squared = result.rsquared_adj # adjusted R²
        f_statistic = result.fvalue  # F-statistic for overall model fit
        f_pvalue = result.f_pvalue   # p-value for the F-statistic
        
        # Sum of squares for meta-analysis
        ss_residual = result.ssr  # sum of squared residuals
        ss_model = result.ess    # explained sum of squares (regression SS)
        ss_total = ss_residual + ss_model  # total variance in the outcome
        
        df_residual = result.df_resid  # degrees of freedom for residuals
        df_model = result.df_model  # degrees of freedom for the model
        
        # Calculate partial eta-squared for group effect
        # Group columns are indices 1 to n_group_cols (after constant)
        partial_eta_sq = self._calculate_partial_eta_squared()
        
        # Log key results
        self.logger.info(f'Coefficients: {coef}')
        self.logger.info(f'Standard Errors: {std_err}')
        self.logger.info(f'P-values: {p_values}')
        self.logger.info(f'R²: {r_squared:.4f}, Adj R²: {adj_r_squared:.4f}')
        self.logger.info(f'F-statistic: {f_statistic:.4f}, p = {f_pvalue:.6f}')
        self.logger.info(f'Partial η² (group effect): {partial_eta_sq:.4f}')
        
        return {
            "sample_size": int(self.sample_size * 0.8),  # training size after split
            "coef_": coef,
            "std_err": std_err,
            "t_values": t_values,
            "p_values": p_values,
            "conf_int_lower": [ci[0] for ci in conf_int],
            "conf_int_upper": [ci[1] for ci in conf_int],
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "f_statistic": f_statistic,
            "f_pvalue": f_pvalue,
            "ss_model": ss_model,
            "ss_residual": ss_residual,
            "ss_total": ss_total,
            "df_model": df_model,
            "df_residual": df_residual,
            "partial_eta_squared": partial_eta_sq,
            "n_group_columns": self.n_group_cols
        }

    def _calculate_partial_eta_squared(self) -> float:
        """
        Calculate partial eta-squared for the group effect.
        partial η² = SS_effect / (SS_effect + SS_error)
        
        For ANCOVA, we need Type III SS which requires fitting reduced models.
        Here we use a simplified approach based on the t-statistics of group coefficients.
        """
        result = self.model_result
        
        # Group coefficients are at indices 1 to n_group_cols (0 is constant)
        group_indices = list(range(1, self.n_group_cols + 1))
        
        if len(group_indices) == 0 or max(group_indices) >= len(result.params):
            return 0.0
        
        # Approximate partial eta-squared using t² / (t² + df_residual)
        # This is valid for single-df effects
        t_squared_sum = sum(result.tvalues[i] ** 2 for i in group_indices if i < len(result.tvalues))
        df_resid = result.df_resid
        
        partial_eta_sq = t_squared_sum / (t_squared_sum + df_resid) if (t_squared_sum + df_resid) > 0 else 0.0
        
        return partial_eta_sq

    def do_aggregate(self) -> bool:
        """
        Aggregate ANCOVA results from all sites using inverse-variance weighted meta-analysis.
        """
        download_mid_artifacts = []
        directory = gen_all_mid_artifacts_url(self.project_id, self.batch_id)
        
        for path in Path(directory).rglob("*-{}-{}-mid-artifacts".format(self.cur_seq, self.get_round())):
            with open(str(path), 'r') as f:
                for line in f:
                    download_mid_artifacts.append(json.loads(line))

        self.logger.debug(f"Downloaded {len(download_mid_artifacts)} mid-artifacts")
        
        if len(download_mid_artifacts) == 0:
            self.logger.warning("No mid-artifacts found for aggregation")
            return False

        # Inverse-variance weighted meta-analysis
        n_coef = len(download_mid_artifacts[0]['coef_'])
        
        pooled_coef = []
        pooled_se = []
        pooled_z = []
        pooled_pvalues = []
        pooled_ci_lower = []
        pooled_ci_upper = []
        
        total_sample_size = sum(a['sample_size'] for a in download_mid_artifacts)
        
        # Pool each coefficient separately
        for i in range(n_coef):
            coefs = [a['coef_'][i] for a in download_mid_artifacts]
            std_errs = [a['std_err'][i] for a in download_mid_artifacts]
            
            # Inverse variance weights
            weights = [1 / (se ** 2) if se > 0 else 0 for se in std_errs]
            total_weight = sum(weights)
            
            if total_weight > 0:
                # Pooled coefficient
                pooled_b = sum(c * w for c, w in zip(coefs, weights)) / total_weight
                # Pooled standard error
                pooled_stderr = np.sqrt(1 / total_weight)
                # Z-score and p-value
                z = pooled_b / pooled_stderr if pooled_stderr > 0 else 0
                p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
                # 95% CI
                ci_lower = pooled_b - 1.96 * pooled_stderr
                ci_upper = pooled_b + 1.96 * pooled_stderr
            else:
                pooled_b = np.mean(coefs)
                pooled_stderr = 0
                z = 0
                p = 1.0
                ci_lower = pooled_b
                ci_upper = pooled_b
            
            pooled_coef.append(pooled_b)
            pooled_se.append(pooled_stderr)
            pooled_z.append(z)
            pooled_pvalues.append(p)
            pooled_ci_lower.append(ci_lower)
            pooled_ci_upper.append(ci_upper)

        # Pool SS components (simple sum for SS, weighted for others)
        total_ss_model = sum(a['ss_model'] for a in download_mid_artifacts)
        total_ss_residual = sum(a['ss_residual'] for a in download_mid_artifacts)
        total_ss_total = total_ss_model + total_ss_residual
        
        total_df_model = download_mid_artifacts[0]['df_model']  # same across sites
        total_df_residual = sum(a['df_residual'] for a in download_mid_artifacts)
        
        # Pooled F-statistic
        ms_model = total_ss_model / total_df_model if total_df_model > 0 else 0
        ms_residual = total_ss_residual / total_df_residual if total_df_residual > 0 else 1
        pooled_f = ms_model / ms_residual if ms_residual > 0 else 0
        pooled_f_pvalue = 1 - scipy_stats.f.cdf(pooled_f, total_df_model, total_df_residual)
        
        # Pooled R²
        pooled_r_squared = total_ss_model / total_ss_total if total_ss_total > 0 else 0
        pooled_adj_r_squared = 1 - (1 - pooled_r_squared) * (total_sample_size - 1) / (total_sample_size - total_df_model - 1)
        
        # Pooled partial eta-squared (weighted average)
        pooled_partial_eta = sum(
            a['partial_eta_squared'] * a['sample_size'] for a in download_mid_artifacts
        ) / total_sample_size if total_sample_size > 0 else 0

        # Log aggregated results
        self.logger.info("=== Aggregated ANCOVA Results ===")
        self.logger.info(f"Total sample size: {total_sample_size}")
        self.logger.info(f"Pooled coefficients: {pooled_coef}")
        self.logger.info(f"Pooled standard errors: {pooled_se}")
        self.logger.info(f"Pooled p-values: {pooled_pvalues}")
        self.logger.info(f"Pooled R²: {pooled_r_squared:.4f}")
        self.logger.info(f"Pooled F-statistic: {pooled_f:.4f}, p = {pooled_f_pvalue:.6f}")
        self.logger.info(f"Pooled partial η²: {pooled_partial_eta:.4f}")

        aggregated_stats = {
            "total_sample_size": total_sample_size,
            "n_sites": len(download_mid_artifacts),
            "coef_": pooled_coef,
            "std_err": pooled_se,
            "z_values": pooled_z,
            "p_values": pooled_pvalues,
            "conf_int_lower": pooled_ci_lower,
            "conf_int_upper": pooled_ci_upper,
            "r_squared": pooled_r_squared,
            "adj_r_squared": pooled_adj_r_squared,
            "f_statistic": pooled_f,
            "f_pvalue": pooled_f_pvalue,
            "ss_model": total_ss_model,
            "ss_residual": total_ss_residual,
            "ss_total": total_ss_total,
            "df_model": total_df_model,
            "df_residual": total_df_residual,
            "partial_eta_squared": pooled_partial_eta
        }

        url = gen_artifacts_url(self.run_id, self.cur_seq, self.get_round())
        self.logger.info(f"Saving aggregated artifacts to: {url}")
        
        if self.save_artifacts(url, json.dumps(aggregated_stats)):
            self.upload(True)
            return True
        
        return False
