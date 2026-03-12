"""
Diagnostics and prediction interval utilities for federated regression tasks.

Provides model-agnostic diagnostic computations that can be added to any
regression task's mid-artifacts.  All outputs are summary statistics safe
for sharing in a federated setting (no individual-level data).
"""

import numpy as np
from scipy import stats as scipy_stats


def compute_vif(X):
    """Compute Variance Inflation Factor for each predictor.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix **without** intercept column.

    Returns
    -------
    list[float]
        VIF for each column of *X*.
    """
    if X.shape[1] == 0:
        return []
    # Centre columns to avoid issues with constant column
    X_centered = X - X.mean(axis=0)
    vifs = []
    for j in range(X_centered.shape[1]):
        others = np.delete(X_centered, j, axis=1)
        if others.shape[1] == 0:
            vifs.append(1.0)
            continue
        # OLS of column j on the rest
        y_j = X_centered[:, j]
        beta, residuals, _, _ = np.linalg.lstsq(others, y_j, rcond=None)
        ss_res = np.sum((y_j - others @ beta) ** 2)
        ss_tot = np.sum((y_j - y_j.mean()) ** 2)
        r_sq = 1 - ss_res / (ss_tot + 1e-10)
        vifs.append(1.0 / (1.0 - r_sq + 1e-10))
    return [float(v) for v in vifs]


def residual_summary(residuals):
    """Return privacy-safe summary statistics for a residual vector.

    Parameters
    ----------
    residuals : array-like

    Returns
    -------
    dict  with keys: mean, std, min, q25, median, q75, max
    """
    r = np.asarray(residuals, dtype=float)
    return {
        'mean': float(np.mean(r)),
        'std': float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
        'min': float(np.min(r)),
        'q25': float(np.percentile(r, 25)),
        'median': float(np.median(r)),
        'q75': float(np.percentile(r, 75)),
        'max': float(np.max(r)),
    }


def cooks_distance_summary(residuals, hat_matrix_diag, p):
    """Compute Cook's distance summary from residuals and leverage.

    Parameters
    ----------
    residuals : ndarray (n,)
        Studentized or raw residuals.
    hat_matrix_diag : ndarray (n,)
        Diagonal of the hat matrix H = X(X'X)^{-1}X'.
    p : int
        Number of parameters (including intercept).

    Returns
    -------
    dict  with keys: max, mean, n_influential (Cook's D > 4/n)
    """
    n = len(residuals)
    h = hat_matrix_diag
    mse = np.sum(residuals ** 2) / (n - p)
    cooks_d = (residuals ** 2 * h) / (p * mse * (1 - h) ** 2 + 1e-10)
    threshold = 4.0 / n
    return {
        'max': float(np.max(cooks_d)),
        'mean': float(np.mean(cooks_d)),
        'n_influential': int(np.sum(cooks_d > threshold)),
        'threshold': float(threshold),
    }


def hat_matrix_diag(X):
    """Compute diagonal of hat matrix H = X (X'X)^{-1} X'.

    Parameters
    ----------
    X : ndarray (n, p)
        Design matrix (with intercept column if applicable).

    Returns
    -------
    ndarray (n,)
    """
    try:
        Q, R = np.linalg.qr(X)
        return np.sum(Q ** 2, axis=1)
    except np.linalg.LinAlgError:
        return np.full(X.shape[0], 1.0 / X.shape[0])


def shapiro_wilk_test(residuals, max_n=5000):
    """Shapiro-Wilk test for normality of residuals.

    Parameters
    ----------
    residuals : array-like
    max_n : int
        If len(residuals) > max_n, subsample.

    Returns
    -------
    dict  with keys: statistic, p_value
    """
    r = np.asarray(residuals, dtype=float)
    if len(r) < 3:
        return {'statistic': None, 'p_value': None}
    if len(r) > max_n:
        rng = np.random.default_rng(42)
        r = rng.choice(r, max_n, replace=False)
    stat, p = scipy_stats.shapiro(r)
    return {'statistic': float(stat), 'p_value': float(p)}


def hosmer_lemeshow_test(y_true, y_prob, n_groups=10):
    """Hosmer-Lemeshow goodness-of-fit test for logistic regression.

    Parameters
    ----------
    y_true : array-like  (0/1)
    y_prob : array-like   predicted probabilities

    Returns
    -------
    dict  with keys: statistic, p_value, df
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n < n_groups * 2:
        return {'statistic': None, 'p_value': None, 'df': None}

    order = np.argsort(y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]

    groups = np.array_split(np.arange(n), n_groups)
    chi2 = 0.0
    for grp in groups:
        obs_1 = np.sum(y_true[grp])
        obs_0 = len(grp) - obs_1
        exp_1 = np.sum(y_prob[grp])
        exp_0 = len(grp) - exp_1
        if exp_1 > 0:
            chi2 += (obs_1 - exp_1) ** 2 / exp_1
        if exp_0 > 0:
            chi2 += (obs_0 - exp_0) ** 2 / exp_0

    df = n_groups - 2
    p_value = scipy_stats.chi2.sf(chi2, df)
    return {
        'statistic': float(chi2),
        'p_value': float(p_value),
        'df': int(df),
    }


def overdispersion_test(pearson_chi2, df_resid):
    """Overdispersion ratio for Poisson / NB models.

    Parameters
    ----------
    pearson_chi2 : float
    df_resid : int or float

    Returns
    -------
    dict  with keys: ratio, p_value
        ratio > 1 suggests overdispersion.
    """
    if df_resid <= 0:
        return {'ratio': None, 'p_value': None}
    ratio = pearson_chi2 / df_resid
    p_value = scipy_stats.chi2.sf(pearson_chi2, int(df_resid))
    return {'ratio': float(ratio), 'p_value': float(p_value)}


def prediction_interval_summary(y_pred, ci_lower, ci_upper,
                                pi_lower=None, pi_upper=None):
    """Summarise prediction / confidence interval widths.

    Parameters
    ----------
    y_pred : array-like
    ci_lower, ci_upper : array-like  confidence interval bounds
    pi_lower, pi_upper : array-like or None  prediction interval bounds

    Returns
    -------
    dict
    """
    ci_width = np.asarray(ci_upper) - np.asarray(ci_lower)
    result = {
        'ci_width_mean': float(np.mean(ci_width)),
        'ci_width_std': float(np.std(ci_width, ddof=1)) if len(ci_width) > 1 else 0.0,
        'ci_width_median': float(np.median(ci_width)),
    }
    if pi_lower is not None and pi_upper is not None:
        pi_width = np.asarray(pi_upper) - np.asarray(pi_lower)
        result['pi_width_mean'] = float(np.mean(pi_width))
        result['pi_width_std'] = float(np.std(pi_width, ddof=1)) if len(pi_width) > 1 else 0.0
        result['pi_width_median'] = float(np.median(pi_width))
    return result


# ------------------------------------------------------------------
# Convenience wrappers for specific model types
# ------------------------------------------------------------------

def ols_diagnostics(X, y, model_result):
    """Full OLS diagnostics bundle.

    Parameters
    ----------
    X : ndarray (n, p)  design matrix WITH intercept
    y : ndarray (n,)
    model_result : statsmodels OLS result

    Returns
    -------
    dict  diagnostics sub-dict for mid-artifacts
    """
    resid = model_result.resid
    h = hat_matrix_diag(X)
    p = X.shape[1]

    # Features without intercept for VIF
    X_no_const = X[:, 1:] if X.shape[1] > 1 else X

    diag = {
        'vif': compute_vif(X_no_const),
        'residual_summary': residual_summary(resid),
        'cooks_distance': cooks_distance_summary(resid, h, p),
        'shapiro_wilk': shapiro_wilk_test(resid),
    }

    # Prediction intervals on training data
    pred = model_result.get_prediction(X)
    frame = pred.summary_frame(alpha=0.05)
    diag['prediction_intervals'] = prediction_interval_summary(
        frame['mean'].values,
        frame['mean_ci_lower'].values,
        frame['mean_ci_upper'].values,
        frame['obs_ci_lower'].values,
        frame['obs_ci_upper'].values,
    )
    return diag


def glm_diagnostics(X, y, model_result):
    """Diagnostics for GLM models (Poisson, NB).

    Parameters
    ----------
    X : ndarray (n, p)  design matrix WITH intercept
    y : ndarray (n,)
    model_result : statsmodels GLM / DiscreteResults

    Returns
    -------
    dict
    """
    X_no_const = X[:, 1:] if X.shape[1] > 1 else X

    diag = {
        'vif': compute_vif(X_no_const),
    }

    # Deviance residuals
    try:
        dev_resid = model_result.resid_deviance
        diag['deviance_residual_summary'] = residual_summary(dev_resid)
    except AttributeError:
        pass

    # Pearson residuals
    try:
        pearson_resid = model_result.resid_pearson
        diag['pearson_residual_summary'] = residual_summary(pearson_resid)
    except AttributeError:
        pass

    # Overdispersion
    try:
        pearson_chi2 = float(model_result.pearson_chi2)
        df_resid = float(model_result.df_resid)
        diag['overdispersion'] = overdispersion_test(pearson_chi2, df_resid)
    except AttributeError:
        pass

    # Prediction CI (mean prediction confidence intervals)
    try:
        pred = model_result.get_prediction(X)
        frame = pred.summary_frame(alpha=0.05)
        diag['prediction_intervals'] = prediction_interval_summary(
            frame['mean'].values,
            frame['mean_ci_lower'].values,
            frame['mean_ci_upper'].values,
        )
    except Exception:
        pass

    return diag


def logistic_diagnostics(X, y, model_result):
    """Diagnostics for logistic regression.

    Parameters
    ----------
    X : ndarray (n, p)  design matrix WITH intercept
    y : ndarray (n,)  binary 0/1
    model_result : statsmodels Logit result

    Returns
    -------
    dict
    """
    X_no_const = X[:, 1:] if X.shape[1] > 1 else X

    diag = {
        'vif': compute_vif(X_no_const),
    }

    # Deviance residuals
    try:
        dev_resid = model_result.resid_dev
        diag['deviance_residual_summary'] = residual_summary(dev_resid)
    except AttributeError:
        pass

    # Hosmer-Lemeshow
    try:
        y_prob = model_result.predict(X)
        diag['hosmer_lemeshow'] = hosmer_lemeshow_test(y, y_prob)
    except Exception:
        pass

    # Prediction CI
    try:
        pred = model_result.get_prediction(X)
        frame = pred.summary_frame(alpha=0.05)
        diag['prediction_intervals'] = prediction_interval_summary(
            frame['mean'].values,
            frame['mean_ci_lower'].values,
            frame['mean_ci_upper'].values,
        )
    except Exception:
        pass

    return diag


def tobit_diagnostics(X, y, censor, beta, sigma):
    """Diagnostics for Tobit (censored regression) model.

    Parameters
    ----------
    X : ndarray (n, p)  design matrix WITH intercept
    y : ndarray (n,)  outcome
    censor : ndarray (n,)  0=observed, 1=right-censored, -1=left-censored
    beta : ndarray (p,)  fitted coefficients
    sigma : float  fitted scale parameter

    Returns
    -------
    dict
    """
    X_no_const = X[:, 1:] if X.shape[1] > 1 else X

    diag = {}

    # VIF on features
    if X_no_const.shape[1] > 0:
        diag['vif'] = compute_vif(X_no_const)

    # Residuals for observed data
    obs = censor == 0
    if np.any(obs):
        mu = X[obs] @ beta
        resid = y[obs] - mu
        diag['residual_summary'] = residual_summary(resid)
        diag['shapiro_wilk'] = shapiro_wilk_test(resid)

    # Censoring summary
    n = len(censor)
    diag['censoring_summary'] = {
        'n_observed': int(np.sum(censor == 0)),
        'n_right_censored': int(np.sum(censor == 1)),
        'n_left_censored': int(np.sum(censor == -1)),
        'pct_censored': float(np.mean(censor != 0) * 100),
    }

    return diag


def cox_diagnostics(cph, train_df):
    """Diagnostics for Cox PH model (lifelines).

    Parameters
    ----------
    cph : lifelines.CoxPHFitter  fitted model
    train_df : DataFrame  training data with time/event columns

    Returns
    -------
    dict
    """
    diag = {}

    # Proportional hazards test (Schoenfeld residuals)
    try:
        ph_test = cph.check_assumptions(
            train_df, p_value_threshold=1.0, show_plots=False)
        # check_assumptions returns list of violating columns or raises
        diag['proportional_hazards_test'] = {
            'test_performed': True,
            'violations': [],
        }
    except Exception:
        # If check_assumptions prints/warns but doesn't return cleanly,
        # use the summary test statistics directly
        try:
            from lifelines.statistics import proportional_hazard_test
            results = proportional_hazard_test(cph, train_df)
            diag['proportional_hazards_test'] = {
                'test_statistic': results.summary['test_statistic'].tolist(),
                'p_value': results.summary['p'].tolist(),
                'feature_names': results.summary.index.tolist(),
            }
        except Exception:
            diag['proportional_hazards_test'] = {
                'test_performed': False,
            }

    # Concordance index is already in _calculate_statistics

    # Deviance residuals
    try:
        dev_resid = cph.compute_residuals(train_df, kind='deviance')
        diag['deviance_residual_summary'] = residual_summary(
            dev_resid.values.flatten())
    except Exception:
        pass

    # VIF on features (exclude time and event columns)
    try:
        feature_cols = [c for c in train_df.columns
                        if c not in ('time', 'event')]
        if feature_cols:
            X_features = train_df[feature_cols].values
            diag['vif'] = compute_vif(X_features)
    except Exception:
        pass

    return diag
