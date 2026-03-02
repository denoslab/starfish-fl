"""
Mixed Effects Logistic Regression (GLMM) for Centralized Analysis

This task performs mixed effects logistic regression (also known as multilevel 
logistic regression or GLMM with binomial family) for binary outcomes with 
hierarchical/clustered data structure.

Mixed effects models include:
- Fixed effects: Variables whose effect is constant across all groups
- Random effects: Variables whose effect varies across groups/clusters

Statistical outputs:
- Fixed Effects: Coefficients (Log-Odds) with standard errors, p-values, confidence intervals
- Random Effects: Variance components (how much groups vary)
- Odds Ratios (exponentiated coefficients)
- Model fit statistics (ELBO, log-likelihood)

Use cases:
- Multi-site clinical trials (patients nested within hospitals)
- Longitudinal studies (repeated measurements within subjects)
- Clustered data (students within schools, employees within companies)
- Federated learning scenarios (data naturally clustered by site)

Model assumptions:
- Binary outcome variable (0/1)
- Independence between clusters (not within)
- Random effects follow normal distribution
- Observations within a cluster may be correlated

Implementation uses Bayesian estimation via Variational Bayes for robust 
uncertainty quantification.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy import sparse
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from starfish.controller.file.file_utils import (
    gen_mid_artifacts_url, gen_all_mid_artifacts_url, gen_artifacts_url,
    downloaded_artifacts_url
)
from starfish.controller.tasks.abstract_task import AbstractTask
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

MIN_SAMPLE_SIZE = 30
MIN_GROUPS = 5
MIN_OBS_PER_GROUP = 3


class MixedEffectsLogisticRegression(AbstractTask):
    """
    Mixed Effects Logistic Regression (GLMM) implementation for centralized analysis.
    
    This model accounts for hierarchical/clustered data structure by including
    random intercepts for groups, allowing each group to have its own baseline
    probability while estimating common fixed effects across all groups.
    
    Expected data format:
    - First column: Group/cluster identifier (e.g., hospital_id, site_id)
    - Middle columns: Predictor variables (X) - continuous or dummy-encoded
    - Last column: Binary outcome variable (Y) - must be 0 or 1
    
    Important notes:
    - Group column should contain categorical identifiers (will be converted to integers)
    - Target variable MUST be binary (0 or 1)
    - Minimum 5 groups recommended for reliable random effect estimates
    - Minimum 3 observations per group recommended
    
    Task config options:
    - total_round: number of rounds (typically 1 for centralized analysis)
    - current_round: current round number
    - vcp_p: Prior standard deviation for variance component parameters (default: 1.0)
    - fe_p: Prior standard deviation for fixed effects parameters (default: 2.0)
    
    Example config:
    {
        "seq": 1,
        "model": "MixedEffectsLogisticRegression",
        "config": {
            "total_round": 1,
            "current_round": 1,
            "vcp_p": 1.0,
            "fe_p": 2.0
        }
    }
    """

    def __init__(self, run):
        super().__init__(run)
        self.sample_size = None
        self.n_groups = None
        self.group_counts = None
        self.group_labels = None
        self.group_mapping = None
        self.X = None
        self.y = None
        self.groups = None
        self.X_test = None
        self.y_test = None
        self.groups_test = None
        self.model_result = None
        self.vcp_p = 1.0  # Prior SD for variance components
        self.fe_p = 2.0   # Prior SD for fixed effects

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
        
        # Get config parameters
        task_config = self.tasks[self.cur_seq - 1].get('config', {})
        self.vcp_p = task_config.get('vcp_p', 1.0)
        self.fe_p = task_config.get('fe_p', 2.0)
        
        self.logger.info(f"Using vcp_p={self.vcp_p}, fe_p={self.fe_p}")
        
        # Extract group column (first column)
        X_array = np.array(X)
        groups_raw = X_array[:, 0]  # First column is group identifier
        X_predictors = X_array[:, 1:]  # Remaining columns are predictors
        
        # Convert group identifiers to integer codes
        unique_groups = np.unique(groups_raw)
        self.n_groups = len(unique_groups)
        self.group_labels = unique_groups.tolist()
        self.group_mapping = {label: idx for idx, label in enumerate(unique_groups)}
        groups_coded = np.array([self.group_mapping[g] for g in groups_raw])
        
        # Validate number of groups
        if self.n_groups < MIN_GROUPS:
            self.logger.warning(
                f"Number of groups ({self.n_groups}) is below recommended minimum ({MIN_GROUPS}). "
                "Random effect estimates may be unreliable."
            )
        
        # Count observations per group
        self.group_counts = {}
        for g in range(self.n_groups):
            count = np.sum(groups_coded == g)
            self.group_counts[self.group_labels[g]] = int(count)
            if count < MIN_OBS_PER_GROUP:
                self.logger.warning(
                    f"Group '{self.group_labels[g]}' has only {count} observations. "
                    f"Minimum {MIN_OBS_PER_GROUP} recommended for stable estimates."
                )
        
        self.logger.info(f"Number of groups: {self.n_groups}")
        self.logger.info(f"Group distribution: {self.group_counts}")
        
        # Validate target is binary
        y_array = np.array(y).flatten()
        unique_targets = np.unique(y_array)
        
        if len(unique_targets) > 2:
            self.logger.error(
                f"Target variable has {len(unique_targets)} unique values. "
                "Mixed Effects Logistic Regression requires binary target (0/1)."
            )
            return False
        
        if not set(unique_targets).issubset({0, 1, 0.0, 1.0}):
            self.logger.warning(
                f"Target values are {unique_targets}. Converting to 0/1..."
            )
            # Map to 0/1
            target_map = {unique_targets[0]: 0, unique_targets[1]: 1}
            y_array = np.array([target_map[v] for v in y_array])
        
        y_array = y_array.astype(int)
        
        # Check outcome distribution per group
        for g_label, g_code in self.group_mapping.items():
            group_mask = groups_coded == g_code
            group_outcomes = y_array[group_mask]
            n_pos = np.sum(group_outcomes == 1)
            n_neg = np.sum(group_outcomes == 0)
            
            if n_pos == 0 or n_neg == 0:
                self.logger.warning(
                    f"Group '{g_label}' has no variation in outcome (all 0s or all 1s). "
                    "This may cause convergence issues."
                )
        
        # Split data - stratify by groups to ensure all groups in both sets
        # For small groups, we may need to handle this carefully
        try:
            X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
                X_predictors, y_array, groups_coded,
                test_size=0.2, random_state=42, stratify=groups_coded
            )
        except ValueError as e:
            self.logger.warning(
                f"Could not stratify by groups (some groups too small): {e}. "
                "Using random split instead."
            )
            X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
                X_predictors, y_array, groups_coded,
                test_size=0.2, random_state=42
            )
        
        self.X = X_train.astype(float)
        self.y = y_train
        self.groups = groups_train
        self.X_test = X_test.astype(float)
        self.y_test = y_test
        self.groups_test = groups_test
        
        self.logger.debug(f'Training data shape: {self.X.shape}')
        self.logger.debug(f'Training label shape: {self.y.shape}')
        self.logger.debug(f'Number of predictors: {self.X.shape[1]}')
        self.logger.debug(f'Groups in training: {len(np.unique(self.groups))}')
        
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

    def _build_random_effects_design(self, groups):
        """
        Build the random effects design matrix for random intercepts.
        
        For random intercepts, we create a sparse matrix where each column
        corresponds to a group, and entries are 1 for observations belonging
        to that group.
        
        Args:
            groups: Array of group indices (0, 1, 2, ..., n_groups-1)
            
        Returns:
            exog_vc: Sparse random effects design matrix
            ident: Array indicating which columns share variance components
        """
        n_obs = len(groups)
        n_groups_in_data = len(np.unique(groups))
        
        # Create sparse random effects design matrix
        # Each column is an indicator for one group
        row_indices = np.arange(n_obs)
        col_indices = groups
        data = np.ones(n_obs)
        
        exog_vc = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_obs, n_groups_in_data)
        )
        
        # ident array: all random intercepts share the same variance component
        # So all columns get the same identifier (0)
        ident = np.zeros(n_groups_in_data, dtype=int)
        
        return exog_vc, ident

    def training(self) -> bool:
        """
        Fit Mixed Effects Logistic Regression model using Variational Bayes.
        """
        self.logger.info('Starting Mixed Effects Logistic Regression analysis...')
        self.logger.info(f'Sample size (training): {len(self.y)}')
        self.logger.info(f'Number of groups: {self.n_groups}')
        self.logger.info(f'Number of predictors: {self.X.shape[1]}')
        
        try:
            # Add intercept to fixed effects
            X_with_intercept = np.column_stack([np.ones(len(self.y)), self.X])
            
            # Build random effects design matrix
            exog_vc, ident = self._build_random_effects_design(self.groups)
            
            self.logger.info(f'Fixed effects design matrix shape: {X_with_intercept.shape}')
            self.logger.info(f'Random effects design matrix shape: {exog_vc.shape}')
            
            # Create variable names
            n_predictors = self.X.shape[1]
            fep_names = ['Intercept'] + [f'X{i+1}' for i in range(n_predictors)]
            vcp_names = ['Group_Intercept_SD']
            vc_names = [f'RE_Group_{self.group_labels[i]}' for i in range(len(np.unique(self.groups)))]
            
            # Fit the model using BinomialBayesMixedGLM
            model = BinomialBayesMixedGLM(
                endog=self.y,
                exog=X_with_intercept,
                exog_vc=exog_vc,
                ident=ident,
                vcp_p=self.vcp_p,
                fe_p=self.fe_p,
                fep_names=fep_names,
                vcp_names=vcp_names,
                vc_names=vc_names
            )
            
            # Fit using Variational Bayes
            self.logger.info("Fitting model using Variational Bayes...")
            self.model_result = model.fit_vb(verbose=False)
            
            self.logger.info('Model fitted successfully.')
            
            # Calculate and save statistics
            stats = self.calculate_statistics()
            
            url = gen_mid_artifacts_url(self.run_id, self.cur_seq, self.get_round())
            self.logger.info(f"Saving mid-artifacts to: {url}")
            
            return self.save_artifacts(url, json.dumps(stats))
            
        except np.linalg.LinAlgError as e:
            self.logger.error(
                f'Linear algebra error during model fitting: {e}. '
                'This may indicate perfect separation, multicollinearity, or singular matrix.'
            )
            return False
        except Exception as e:
            self.logger.error(f'Error during Mixed Effects Logistic Regression training: {e}')
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def calculate_statistics(self) -> dict:
        """
        Calculate Mixed Effects Logistic Regression statistics.
        
        The BinomialBayesMixedGLM result contains:
        - Fixed effects parameters (first n_fixed parameters)
        - Variance component parameters (log scale)
        - Random effects realizations
        """
        result = self.model_result
        
        # Number of fixed effects (including intercept)
        n_fixed = self.X.shape[1] + 1  # +1 for intercept
        
        # Extract fixed effects statistics
        # In VB results, params contains: [fixed_effects, variance_components, random_effects]
        fe_mean = result.fe_mean.tolist()  # Fixed effects posterior means
        fe_sd = result.fe_sd.tolist()      # Fixed effects posterior SDs
        
        # Calculate z-scores and p-values for fixed effects
        fe_z_values = []
        fe_p_values = []
        fe_ci_lower = []
        fe_ci_upper = []
        
        for i in range(n_fixed):
            mean = fe_mean[i]
            sd = fe_sd[i]
            
            # Z-score
            z = mean / sd if sd > 0 else 0
            fe_z_values.append(z)
            
            # Two-tailed p-value
            p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
            fe_p_values.append(p)
            
            # 95% CI (using normal approximation)
            ci_lower = mean - 1.96 * sd
            ci_upper = mean + 1.96 * sd
            fe_ci_lower.append(ci_lower)
            fe_ci_upper.append(ci_upper)
        
        # Calculate Odds Ratios (exp of coefficients)
        odds_ratios = np.exp(fe_mean).tolist()
        odds_ratio_ci_lower = np.exp(fe_ci_lower).tolist()
        odds_ratio_ci_upper = np.exp(fe_ci_upper).tolist()
        
        # Variance components (random effects)
        # vcp_mean is the posterior mean of log(SD), so exp(vcp_mean) gives SD
        vcp_mean = result.vcp_mean.tolist()  # Log scale
        vcp_sd = result.vcp_sd.tolist()      # SD of log scale
        
        # Convert to SD scale
        random_effect_sd = np.exp(vcp_mean).tolist()
        random_effect_var = (np.exp(vcp_mean) ** 2).tolist()
        
        # Random effects realizations (group-specific intercepts)
        vc_mean = result.vc_mean.tolist()  # Random effect means
        vc_sd = result.vc_sd.tolist()      # Random effect SDs
        
        # Create group-level random effects dictionary
        groups_in_training = np.unique(self.groups)
        random_intercepts = {}
        for i, g_idx in enumerate(groups_in_training):
            g_label = self.group_labels[g_idx]
            if i < len(vc_mean):
                random_intercepts[str(g_label)] = {
                    'mean': vc_mean[i],
                    'sd': vc_sd[i]
                }
        
        # Model fit statistics
        # ELBO (Evidence Lower Bound) - higher is better
        # Note: We need to calculate this as it's not always directly available
        try:
            elbo = float(result.model.vb_elbo(result.params[:len(fe_mean) + len(vcp_mean) + len(vc_mean)],
                                               result.params[len(fe_mean) + len(vcp_mean) + len(vc_mean):]))
        except:
            elbo = None
        
        # Calculate ICC (Intraclass Correlation Coefficient)
        # ICC = var(random intercept) / (var(random intercept) + pi^2/3)
        # pi^2/3 ≈ 3.29 is the residual variance for logistic models
        if len(random_effect_var) > 0:
            icc = random_effect_var[0] / (random_effect_var[0] + (np.pi**2 / 3))
        else:
            icc = 0.0
        
        # Log key results
        self.logger.info("=== Mixed Effects Logistic Regression Results ===")
        self.logger.info(f"Sample size (training): {int(self.sample_size * 0.8)}")
        self.logger.info(f"Number of groups: {self.n_groups}")
        self.logger.info(f"Fixed Effects (log-odds): {fe_mean}")
        self.logger.info(f"Fixed Effects SDs: {fe_sd}")
        self.logger.info(f"Odds Ratios: {odds_ratios}")
        self.logger.info(f"P-values: {fe_p_values}")
        self.logger.info(f"Random Effect SD: {random_effect_sd}")
        self.logger.info(f"ICC: {icc:.4f}")
        
        return {
            # Sample information
            "sample_size": int(self.sample_size * 0.8),
            "n_groups": self.n_groups,
            "group_counts": self.group_counts,
            "group_labels": [str(g) for g in self.group_labels],
            
            # Fixed effects (main results)
            "fe_coef": fe_mean,
            "fe_std_err": fe_sd,
            "fe_z_values": fe_z_values,
            "fe_p_values": fe_p_values,
            "fe_conf_int_lower": fe_ci_lower,
            "fe_conf_int_upper": fe_ci_upper,
            
            # Odds ratios (easier interpretation)
            "odds_ratios": odds_ratios,
            "odds_ratio_ci_lower": odds_ratio_ci_lower,
            "odds_ratio_ci_upper": odds_ratio_ci_upper,
            
            # Variance components (random effects structure)
            "vcp_mean_log": vcp_mean,  # Log scale
            "vcp_sd_log": vcp_sd,
            "random_effect_sd": random_effect_sd,  # SD scale
            "random_effect_var": random_effect_var,  # Variance scale
            
            # Group-level random intercepts
            "random_intercepts": random_intercepts,
            
            # Model fit statistics
            "icc": icc,  # Intraclass correlation coefficient
            
            # Prior settings used
            "vcp_p": self.vcp_p,
            "fe_p": self.fe_p
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
                "Mixed Effects Logistic Regression is configured for centralized (single-site) analysis. "
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

        self.logger.info("=== Final Mixed Effects Logistic Regression Results (Centralized) ===")
        self.logger.info(f"Sample size: {final_stats['sample_size']}")
        self.logger.info(f"Number of groups: {final_stats['n_groups']}")
        self.logger.info(f"Fixed Effects (log-odds): {final_stats['fe_coef']}")
        self.logger.info(f"Odds Ratios: {final_stats['odds_ratios']}")
        self.logger.info(f"P-values: {final_stats['fe_p_values']}")
        self.logger.info(f"Random Effect SD: {final_stats['random_effect_sd']}")
        self.logger.info(f"ICC: {final_stats['icc']:.4f}")

        url = gen_artifacts_url(self.run_id, self.cur_seq, self.get_round())
        self.logger.info(f"Saving final artifacts to: {url}")
        
        if self.save_artifacts(url, json.dumps(final_stats)):
            self.upload(True)
            return True
        
        return False
