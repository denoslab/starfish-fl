import os

from starfish.controller.tasks.abstract_r_task import AbstractRTask


class RCensoredRegression(AbstractRTask):
    """
    Federated censored regression (Tobit Type I) implemented in R.

    Uses R's survival::survreg(dist="gaussian") for local fitting and
    inverse-variance weighted meta-analysis for aggregation.
    """

    def __init__(self, run):
        self.r_script_dir = os.path.join(
            os.path.dirname(__file__), 'scripts')
        super().__init__(run)
