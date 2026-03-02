"""
Statistical Logistic Regression for Federated Learning

This task performs logistic regression for binary classification with a focus on 
statistical inference (coefficients, p-values, confidence intervals) rather than 
just prediction.

Statistical outputs:
- Coefficients (Log-Odds) with standard errors, p-values, confidence intervals
- Odds Ratios (exponentiated coefficients)
- Pseudo R-squared (McFadden's)
- Likelihood Ratio Chi-Squared statistic

Federated approach:
- Each site computes local Logistic Regression using Maximum Likelihood Estimation (MLE)
- Sites share: coefficients, standard errors, sample size, model fit statistics
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


class LogisticRegressionStats(AbstractTask):
    """
    Statistical Logistic Regression implementation for federated analysis.
    
    Expected data format:
    - Last column: Binary outcome variable (Y) - must be 0 or 1
    - Remaining columns: Predictor variables (X) - continuous or dummy-encoded
    
    Task config should specify:
    - total_round: number of FL rounds (typically 1 for statistical meta-analysis)
    - current_round: current round number
    """

    def __init__(self, run):
        super().__init__(run)
        self.sample_size = None
        self.X = None
        self.y = None
        self.X_with_const = None
        self.model_result = None

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
        
        # Ensure target is binary
        unique_targets = np.unique(y)
        if len(unique_targets) > 2:
            self.logger.error(f"Target variable has {len(unique_targets)} classes. Logistic Regression requires binary target.")
            return False
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Add constant for statsmodels (intercept)
        self.X_with_const = sm.add_constant(self.X)
        
        self.logger.debug(f'Training data shape: {self.X.shape}')
        self.logger.debug(f'Training label shape: {self.y.shape}')
        
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
        Fit Logistic Regression model and compute statistics.
        """
        self.logger.info('Starting Statistical Logistic Regression analysis...')
        
        try:
            # Fit Logit model
            model = sm.Logit(self.y, self.X_with_const)
            
            # Use disp=0 to suppress convergence output
            self.model_result = model.fit(disp=0)
            
            self.logger.info(f'Model fitted. Pseudo R² = {self.model_result.prsquared:.4f}')
            
            # Calculate and save statistics
            stats = self.calculate_statistics()
            
            url = gen_mid_artifacts_url(self.run_id, self.cur_seq, self.get_round())
            self.logger.info(f"Saving mid-artifacts to: {url}")
            
            return self.save_artifacts(url, json.dumps(stats))
            
        except Exception as e:
            self.logger.error(f'Error during Logistic Regression training: {e}')
            return False

    def calculate_statistics(self) -> dict:
        """
        Calculate Logistic Regression statistics for federated aggregation.
        """
        result = self.model_result
        
        # Basic coefficients and inference
        coef = result.params.tolist()
        std_err = result.bse.tolist()
        z_values = result.tvalues.tolist() # In statsmodels Logit, tvalues are z-scores
        p_values = result.pvalues.tolist()
        conf_int = result.conf_int().tolist()
        
        # Odds Ratios (OR)
        odds_ratios = np.exp(coef).tolist()   # It's the coefficient transformed from the log-odds scale back to an understandable odds scale ($e^{\beta}$).
        
        # Model fit statistics
        prsquared = result.prsquared # McFadden's Pseudo R-squared
        llr = result.llr # Likelihood Ratio Chi-Squared statistic
        llr_pvalue = result.llr_pvalue
        
        # Log-Likelihoods
        llf = result.llf # Log-likelihood of full model
        llnull = result.llnull # Log-likelihood of null model (intercept only)
        
        # Log key results
        self.logger.info(f'Coefficients: {coef}')
        self.logger.info(f'Odds Ratios: {odds_ratios}')
        self.logger.info(f'P-values: {p_values}')
        self.logger.info(f'Pseudo R²: {prsquared:.4f}')
        self.logger.info(f'LLR p-value: {llr_pvalue:.6f}')
        
        return {
            "sample_size": int(self.sample_size * 0.8),
            "coef_": coef,
            "std_err": std_err,
            "z_values": z_values,
            "p_values": p_values,
            "conf_int_lower": [ci[0] for ci in conf_int],
            "conf_int_upper": [ci[1] for ci in conf_int],
            "odds_ratios": odds_ratios,
            "prsquared": prsquared,
            "llr": llr,
            "llr_pvalue": llr_pvalue,
            "llf": llf,
            "llnull": llnull
        }

    def do_aggregate(self) -> bool:
        """
        Aggregate Logistic Regression results from all sites using inverse-variance weighted meta-analysis.
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

        # Calculate pooled Odds Ratios
        pooled_odds_ratios = np.exp(pooled_coef).tolist()

        # Pool Model Fit Statistics
        # For Pseudo R2 and LLR, we can weight by sample size as an approximation
        pooled_prsquared = sum(
            a['prsquared'] * a['sample_size'] for a in download_mid_artifacts
        ) / total_sample_size if total_sample_size > 0 else 0
        
        # Summing Log-Likelihoods is valid if we assume independence between sites
        total_llf = sum(a['llf'] for a in download_mid_artifacts)
        total_llnull = sum(a['llnull'] for a in download_mid_artifacts)
        
        # Recalculate LLR based on summed log-likelihoods
        # LLR = -2 * (llnull - llf)
        pooled_llr = -2 * (total_llnull - total_llf)
        
        # Degrees of freedom for LLR is number of predictors (excluding intercept)
        df_model = n_coef - 1 
        pooled_llr_pvalue = scipy_stats.chi2.sf(pooled_llr, df_model)

        # Log aggregated results
        self.logger.info("=== Aggregated Logistic Regression Results ===")
        self.logger.info(f"Total sample size: {total_sample_size}")
        self.logger.info(f"Pooled coefficients: {pooled_coef}")
        self.logger.info(f"Pooled Odds Ratios: {pooled_odds_ratios}")
        self.logger.info(f"Pooled p-values: {pooled_pvalues}")
        self.logger.info(f"Pooled Pseudo R²: {pooled_prsquared:.4f}")
        self.logger.info(f"Pooled LLR p-value: {pooled_llr_pvalue:.6f}")

        aggregated_stats = {
            "total_sample_size": total_sample_size,
            "n_sites": len(download_mid_artifacts),
            "coef_": pooled_coef,
            "std_err": pooled_se,
            "z_values": pooled_z,
            "p_values": pooled_pvalues,
            "conf_int_lower": pooled_ci_lower,
            "conf_int_upper": pooled_ci_upper,
            "odds_ratios": pooled_odds_ratios,
            "prsquared": pooled_prsquared,
            "llr": pooled_llr,
            "llr_pvalue": pooled_llr_pvalue,
            "llf": total_llf,
            "llnull": total_llnull
        }

        url = gen_artifacts_url(self.run_id, self.cur_seq, self.get_round())
        self.logger.info(f"Saving aggregated artifacts to: {url}")
        
        if self.save_artifacts(url, json.dumps(aggregated_stats)):
            self.upload(True)
            return True
        
        return False
