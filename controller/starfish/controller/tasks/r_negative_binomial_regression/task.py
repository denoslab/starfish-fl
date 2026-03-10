import os

from starfish.controller.tasks.abstract_r_task import AbstractRTask


class RNegativeBinomialRegression(AbstractRTask):
    """
    Federated Negative Binomial Regression implemented in R.

    Uses R's MASS::glm.nb() for local fitting and inverse-variance
    weighted meta-analysis for aggregation.
    """

    def __init__(self, run):
        self.r_script_dir = os.path.join(
            os.path.dirname(__file__), 'scripts')
        super().__init__(run)
