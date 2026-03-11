import numpy as np
from django.test import TestCase

from starfish.controller.tasks.diagnostics import (
    compute_vif, residual_summary, cooks_distance_summary,
    hat_matrix_diag, shapiro_wilk_test, hosmer_lemeshow_test,
    overdispersion_test, prediction_interval_summary,
    ols_diagnostics, glm_diagnostics, logistic_diagnostics,
)


class VIFTest(TestCase):
    def test_orthogonal_features_have_vif_near_one(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 3))
        vifs = compute_vif(X)
        self.assertEqual(len(vifs), 3)
        for v in vifs:
            self.assertAlmostEqual(v, 1.0, delta=0.3)

    def test_collinear_features_have_high_vif(self):
        rng = np.random.default_rng(42)
        x1 = rng.standard_normal(200)
        x2 = x1 + rng.normal(0, 0.01, 200)  # nearly identical
        x3 = rng.standard_normal(200)
        X = np.column_stack([x1, x2, x3])
        vifs = compute_vif(X)
        self.assertGreater(vifs[0], 10)
        self.assertGreater(vifs[1], 10)

    def test_single_feature(self):
        X = np.random.default_rng(42).standard_normal((100, 1))
        vifs = compute_vif(X)
        self.assertEqual(len(vifs), 1)
        self.assertAlmostEqual(vifs[0], 1.0, delta=0.01)

    def test_empty_features(self):
        X = np.empty((100, 0))
        self.assertEqual(compute_vif(X), [])


class ResidualSummaryTest(TestCase):
    def test_basic_summary(self):
        r = np.array([1.0, -1.0, 0.5, -0.5, 0.0])
        s = residual_summary(r)
        self.assertIn('mean', s)
        self.assertIn('std', s)
        self.assertIn('min', s)
        self.assertIn('median', s)
        self.assertIn('max', s)
        self.assertAlmostEqual(s['mean'], 0.0, places=5)
        self.assertEqual(s['min'], -1.0)
        self.assertEqual(s['max'], 1.0)

    def test_single_value(self):
        s = residual_summary([3.0])
        self.assertEqual(s['mean'], 3.0)
        self.assertEqual(s['std'], 0.0)


class HatMatrixTest(TestCase):
    def test_hat_diagonal_sums_to_p(self):
        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(50), rng.standard_normal((50, 2))])
        h = hat_matrix_diag(X)
        self.assertEqual(len(h), 50)
        self.assertAlmostEqual(np.sum(h), 3.0, places=5)


class CooksDistanceTest(TestCase):
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(42)
        r = rng.standard_normal(100)
        h = np.full(100, 0.03)
        result = cooks_distance_summary(r, h, 3)
        self.assertIn('max', result)
        self.assertIn('mean', result)
        self.assertIn('n_influential', result)
        self.assertIn('threshold', result)
        self.assertAlmostEqual(result['threshold'], 4.0 / 100)


class ShapiroWilkTest(TestCase):
    def test_normal_data(self):
        rng = np.random.default_rng(42)
        r = rng.standard_normal(100)
        result = shapiro_wilk_test(r)
        self.assertIsNotNone(result['statistic'])
        self.assertGreater(result['p_value'], 0.05)

    def test_non_normal_data(self):
        r = np.concatenate([np.zeros(50), np.ones(50) * 100])
        result = shapiro_wilk_test(r)
        self.assertIsNotNone(result['statistic'])
        self.assertLess(result['p_value'], 0.05)

    def test_too_few_samples(self):
        result = shapiro_wilk_test([1.0, 2.0])
        self.assertIsNone(result['statistic'])


class HosmerLemeshowTest(TestCase):
    def test_well_calibrated_model(self):
        rng = np.random.default_rng(42)
        y_prob = rng.uniform(0, 1, 200)
        y_true = (rng.uniform(0, 1, 200) < y_prob).astype(float)
        result = hosmer_lemeshow_test(y_true, y_prob)
        self.assertIn('statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('df', result)

    def test_too_few_samples(self):
        result = hosmer_lemeshow_test([0, 1], [0.3, 0.7])
        self.assertIsNone(result['statistic'])


class OverdispersionTest(TestCase):
    def test_no_overdispersion(self):
        result = overdispersion_test(100.0, 100)
        self.assertAlmostEqual(result['ratio'], 1.0)

    def test_overdispersion(self):
        result = overdispersion_test(300.0, 100)
        self.assertGreater(result['ratio'], 1.0)

    def test_zero_df(self):
        result = overdispersion_test(10.0, 0)
        self.assertIsNone(result['ratio'])


class PredictionIntervalSummaryTest(TestCase):
    def test_ci_only(self):
        pred = np.array([1.0, 2.0, 3.0])
        ci_l = np.array([0.5, 1.5, 2.5])
        ci_u = np.array([1.5, 2.5, 3.5])
        result = prediction_interval_summary(pred, ci_l, ci_u)
        self.assertAlmostEqual(result['ci_width_mean'], 1.0)
        self.assertNotIn('pi_width_mean', result)

    def test_with_pi(self):
        pred = np.array([1.0, 2.0, 3.0])
        ci_l = np.array([0.5, 1.5, 2.5])
        ci_u = np.array([1.5, 2.5, 3.5])
        pi_l = np.array([0.0, 1.0, 2.0])
        pi_u = np.array([2.0, 3.0, 4.0])
        result = prediction_interval_summary(pred, ci_l, ci_u, pi_l, pi_u)
        self.assertAlmostEqual(result['ci_width_mean'], 1.0)
        self.assertAlmostEqual(result['pi_width_mean'], 2.0)


class OLSDiagnosticsIntegrationTest(TestCase):
    def test_full_ols_diagnostics(self):
        import statsmodels.api as sm
        rng = np.random.default_rng(42)
        X_raw = rng.standard_normal((100, 3))
        X = sm.add_constant(X_raw)
        y = X_raw @ [1.0, -0.5, 0.3] + rng.normal(0, 0.5, 100)
        model = sm.OLS(y, X).fit()
        diag = ols_diagnostics(X, y, model)
        self.assertIn('vif', diag)
        self.assertEqual(len(diag['vif']), 3)
        self.assertIn('residual_summary', diag)
        self.assertIn('cooks_distance', diag)
        self.assertIn('shapiro_wilk', diag)
        self.assertIn('prediction_intervals', diag)
        self.assertIn('pi_width_mean', diag['prediction_intervals'])


class GLMDiagnosticsIntegrationTest(TestCase):
    def test_poisson_diagnostics(self):
        import statsmodels.api as sm
        rng = np.random.default_rng(42)
        X_raw = rng.standard_normal((200, 2))
        X = sm.add_constant(X_raw)
        mu = np.exp(X_raw @ [0.3, -0.2])
        y = rng.poisson(mu)
        model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        diag = glm_diagnostics(X, y, model)
        self.assertIn('vif', diag)
        self.assertEqual(len(diag['vif']), 2)
        self.assertIn('deviance_residual_summary', diag)
        self.assertIn('overdispersion', diag)


class LogisticDiagnosticsIntegrationTest(TestCase):
    def test_logistic_diagnostics(self):
        import statsmodels.api as sm
        rng = np.random.default_rng(42)
        X_raw = rng.standard_normal((200, 2))
        X = sm.add_constant(X_raw)
        prob = 1 / (1 + np.exp(-(X_raw @ [1.0, -0.5])))
        y = rng.binomial(1, prob)
        model = sm.Logit(y, X).fit(disp=0)
        diag = logistic_diagnostics(X, y, model)
        self.assertIn('vif', diag)
        self.assertIn('hosmer_lemeshow', diag)
        self.assertIn('deviance_residual_summary', diag)
