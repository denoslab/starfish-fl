"""
Ordinal Logistic Regression (Proportional Odds Model) for Centralized Analysis

This task performs ordinal logistic regression for ordered categorical outcomes 
with a focus on statistical inference (coefficients, p-values, confidence intervals).

Statistical outputs:
- Coefficients (Log-Odds) with standard errors, p-values, confidence intervals
- Threshold/Cut-point parameters (for K categories, K-1 thresholds)
- Odds Ratios (exponentiated coefficients)
- Pseudo R-squared (McFadden's)
- Likelihood Ratio Chi-Squared statistic

Model assumptions:
- Proportional Odds: The effect of predictors is constant across all thresholds
- Independence of observations
- No multicollinearity among predictors
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

from starfish.controller.file.file_utils import (
    gen_mid_artifacts_url, gen_all_mid_artifacts_url, gen_artifacts_url,
    downloaded_artifacts_url
)
from starfish.controller.tasks.abstract_task import AbstractTask
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

MIN_SAMPLE_SIZE = 30
MIN_CATEGORIES = 3


class OrdinalLogisticRegression(AbstractTask):
    """
    Ordinal Logistic Regression (Proportional Odds Model) implementation for centralized analysis.
    
    Expected data format:
    - Last column: Ordinal outcome variable (Y) - must be integer-coded (0, 1, 2, ..., K-1) 
                   representing ordered categories
    - Remaining columns: Predictor variables (X) - continuous or dummy-encoded
    
    Important notes:
    - Target variable MUST be pre-encoded as integers (0, 1, 2, ...)
    - Categories must be naturally ordered (0 < 1 < 2 < ...)
    - Minimum 3 categories required (for 2 categories, use binary logistic regression)
    
    Task config should specify:
    - total_round: number of rounds (typically 1 for centralized analysis)
    - current_round: current round number
    """

    def __init__(self, run):
        super().__init__(run)
        self.sample_size = None
        self.n_categories = None
        self.category_counts = None
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
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
        
        # Validate target variable is integer-coded
        y_array = np.array(y).flatten()
        
        if not np.issubdtype(y_array.dtype, np.integer):
            # Try to convert to integer
            try:
                y_array = y_array.astype(int)
                self.logger.info("Converted target variable to integer type")
            except (ValueError, TypeError):
                self.logger.error(
                    "Target variable must be integer-coded (0, 1, 2, ...). "
                    "Please pre-encode your ordinal categories as integers."
                )
                return False
        
        # Get unique categories and validate
        unique_categories = np.unique(y_array)
        self.n_categories = len(unique_categories)
        
        # Check minimum categories
        if self.n_categories < MIN_CATEGORIES:
            self.logger.error(
                f"Target variable has only {self.n_categories} categories. "
                f"Ordinal Logistic Regression requires at least {MIN_CATEGORIES} ordered categories. "
                "For binary outcomes (2 categories), please use Binary Logistic Regression instead."
            )
            return False
        
        # Check that categories are consecutive integers starting from 0
        expected_categories = np.arange(self.n_categories)
        if not np.array_equal(unique_categories, expected_categories):
            self.logger.warning(
                f"Categories should be consecutive integers starting from 0. "
                f"Found: {unique_categories.tolist()}. Expected: {expected_categories.tolist()}. "
                f"Remapping categories..."
            )
            # Remap to consecutive integers
            category_map = {old: new for new, old in enumerate(unique_categories)}
            y_array = np.array([category_map[val] for val in y_array])
        
        # Count observations per category
        self.category_counts = {}
        for cat in range(self.n_categories):
            count = np.sum(y_array == cat)
            self.category_counts[cat] = int(count)
            if count < 5:
                self.logger.warning(
                    f"Category {cat} has only {count} observations. "
                    "This may cause convergence issues or unreliable estimates."
                )
        
        self.logger.info(f"Number of ordinal categories: {self.n_categories}")
        self.logger.info(f"Category distribution: {self.category_counts}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_array, test_size=0.2, random_state=42, stratify=y_array
        )
        
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.logger.debug(f'Training data shape: {self.X.shape}')
        self.logger.debug(f'Training label shape: {self.y.shape}')
        self.logger.debug(f'Test data shape: {self.X_test.shape}')
        
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
        Fit Ordinal Logistic Regression model and compute statistics.
        """
        self.logger.info('Starting Ordinal Logistic Regression (Proportional Odds Model) analysis...')
        self.logger.info(f'Number of categories: {self.n_categories}')
        self.logger.info(f'Number of predictors: {self.X.shape[1]}')
        
        try:
            # Fit Ordered Logit model using statsmodels
            # distr='logit' specifies the proportional odds (cumulative logit) model
            model = OrderedModel(self.y, self.X, distr='logit')
            
            # Fit with MLE, disp=0 suppresses convergence output
            self.model_result = model.fit(method='bfgs', disp=0)
            
            self.logger.info(f'Model fitted successfully.')
            self.logger.info(f'Pseudo R² (McFadden) = {self.model_result.prsquared:.4f}')
            
            # Calculate and save statistics
            stats = self.calculate_statistics()
            
            url = gen_mid_artifacts_url(self.run_id, self.cur_seq, self.get_round())
            self.logger.info(f"Saving mid-artifacts to: {url}")
            
            return self.save_artifacts(url, json.dumps(stats))
            
        except np.linalg.LinAlgError as e:
            self.logger.error(
                f'Linear algebra error during model fitting: {e}. '
                'This may indicate perfect separation or multicollinearity.'
            )
            return False
        except Exception as e:
            self.logger.error(f'Error during Ordinal Logistic Regression training: {e}')
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def calculate_statistics(self) -> dict:
        """
        Calculate Ordinal Logistic Regression statistics.
        
        The OrderedModel in statsmodels returns:
        - First (n_predictors) parameters: coefficients for predictors
        - Last (n_categories - 1) parameters: threshold/cut-point parameters
        """
        result = self.model_result
        
        # Get number of predictors and thresholds
        n_predictors = self.X.shape[1]
        n_thresholds = self.n_categories - 1
        
        # Extract all parameters
        all_params = result.params.tolist()
        all_std_err = result.bse.tolist()
        all_z_values = result.tvalues.tolist()  # z-scores for MLE
        all_p_values = result.pvalues.tolist()
        all_conf_int = result.conf_int().tolist()
        
        # Split into coefficients and thresholds
        # In statsmodels OrderedModel: first n_predictors are betas, last n_thresholds are cut-points
        coef = all_params[:n_predictors]
        coef_std_err = all_std_err[:n_predictors]
        coef_z_values = all_z_values[:n_predictors]
        coef_p_values = all_p_values[:n_predictors]
        coef_conf_int_lower = [ci[0] for ci in all_conf_int[:n_predictors]]
        coef_conf_int_upper = [ci[1] for ci in all_conf_int[:n_predictors]]
        
        thresholds = all_params[n_predictors:]
        threshold_std_err = all_std_err[n_predictors:]
        threshold_z_values = all_z_values[n_predictors:]
        threshold_p_values = all_p_values[n_predictors:]
        threshold_conf_int_lower = [ci[0] for ci in all_conf_int[n_predictors:]]
        threshold_conf_int_upper = [ci[1] for ci in all_conf_int[n_predictors:]]
        
        # Calculate Odds Ratios for coefficients (exp(beta))
        odds_ratios = np.exp(coef).tolist()
        odds_ratio_ci_lower = np.exp(coef_conf_int_lower).tolist()
        odds_ratio_ci_upper = np.exp(coef_conf_int_upper).tolist()
        
        # Model fit statistics
        prsquared = result.prsquared  # McFadden's Pseudo R-squared
        llf = result.llf  # Log-likelihood of full model
        llnull = result.llnull  # Log-likelihood of null model
        
        # Likelihood Ratio Chi-Squared: LLR = -2 * (llnull - llf)
        llr = -2 * (llnull - llf)
        # Degrees of freedom = number of predictors
        llr_df = n_predictors
        llr_pvalue = scipy_stats.chi2.sf(llr, llr_df)
        
        # AIC and BIC for model comparison
        aic = result.aic
        bic = result.bic
        
        # Log key results
        self.logger.info("=== Ordinal Logistic Regression Results ===")
        self.logger.info(f'Sample size (training): {int(self.sample_size * 0.8)}')
        self.logger.info(f'Number of categories: {self.n_categories}')
        self.logger.info(f'Coefficients (log-odds): {coef}')
        self.logger.info(f'Odds Ratios: {odds_ratios}')
        self.logger.info(f'P-values (coefficients): {coef_p_values}')
        self.logger.info(f'Thresholds (cut-points): {thresholds}')
        self.logger.info(f'Pseudo R² (McFadden): {prsquared:.4f}')
        self.logger.info(f'LLR Chi²: {llr:.4f}, df={llr_df}, p={llr_pvalue:.6f}')
        self.logger.info(f'AIC: {aic:.4f}, BIC: {bic:.4f}')
        
        return {
            # Sample information
            "sample_size": int(self.sample_size * 0.8),
            "n_categories": self.n_categories,
            "category_counts": self.category_counts,
            
            # Predictor coefficients (main results)
            "coef_": coef,
            "std_err": coef_std_err,
            "z_values": coef_z_values,
            "p_values": coef_p_values,
            "conf_int_lower": coef_conf_int_lower,
            "conf_int_upper": coef_conf_int_upper,
            
            # Odds ratios (easier interpretation)
            "odds_ratios": odds_ratios,
            "odds_ratio_ci_lower": odds_ratio_ci_lower,
            "odds_ratio_ci_upper": odds_ratio_ci_upper,
            
            # Threshold parameters
            "thresholds": thresholds,
            "threshold_std_err": threshold_std_err,
            "threshold_z_values": threshold_z_values,
            "threshold_p_values": threshold_p_values,
            "threshold_conf_int_lower": threshold_conf_int_lower,
            "threshold_conf_int_upper": threshold_conf_int_upper,
            
            # Model fit statistics
            "prsquared": prsquared,
            "llf": llf,
            "llnull": llnull,
            "llr": llr,
            "llr_df": llr_df,
            "llr_pvalue": llr_pvalue,
            "aic": aic,
            "bic": bic
        }

    def do_aggregate(self) -> bool:
        """
        For centralized version: Simply collect the single site's results and save as final artifacts.
        No aggregation across multiple sites is performed.
        """
        download_mid_artifacts = []
        directory = gen_all_mid_artifacts_url(self.project_id, self.batch_id)
        
        for path in Path(directory).rglob("*-{}-{}-mid-artifacts".format(self.cur_seq, self.get_round())):
            with open(str(path), 'r') as f:
                for line in f:
                    download_mid_artifacts.append(json.loads(line))

        self.logger.debug(f"Downloaded {len(download_mid_artifacts)} mid-artifacts")
        
        if len(download_mid_artifacts) == 0:
            self.logger.warning("No mid-artifacts found")
            return False
        
        if len(download_mid_artifacts) > 1:
            self.logger.warning(
                f"Found {len(download_mid_artifacts)} site results. "
                "Ordinal Logistic Regression is configured for centralized (single-site) analysis. "
                "Using the first site's results only."
            )
        
        # For centralized analysis, use the single site's results directly
        site_stats = download_mid_artifacts[0]
        
        # Add metadata to indicate this is a centralized analysis
        final_stats = {
            "analysis_type": "centralized",
            "n_sites": 1,
            **site_stats
        }

        self.logger.info("=== Final Ordinal Logistic Regression Results (Centralized) ===")
        self.logger.info(f"Sample size: {final_stats['sample_size']}")
        self.logger.info(f"Number of categories: {final_stats['n_categories']}")
        self.logger.info(f"Coefficients: {final_stats['coef_']}")
        self.logger.info(f"Odds Ratios: {final_stats['odds_ratios']}")
        self.logger.info(f"P-values: {final_stats['p_values']}")
        self.logger.info(f"Pseudo R²: {final_stats['prsquared']:.4f}")
        self.logger.info(f"LLR Chi² p-value: {final_stats['llr_pvalue']:.6f}")

        url = gen_artifacts_url(self.run_id, self.cur_seq, self.get_round())
        self.logger.info(f"Saving final artifacts to: {url}")
        
        if self.save_artifacts(url, json.dumps(final_stats)):
            self.upload(True)
            return True
        
        return False
