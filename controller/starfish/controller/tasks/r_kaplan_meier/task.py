import os

from starfish.controller.tasks.abstract_r_task import AbstractRTask


class RKaplanMeier(AbstractRTask):
    """
    Federated Kaplan-Meier estimation with log-rank test, implemented in R.

    Uses R's survival::survfit() and survival::survdiff().
    """

    def __init__(self, run):
        self.r_script_dir = os.path.join(
            os.path.dirname(__file__), 'scripts')
        super().__init__(run)
