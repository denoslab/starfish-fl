import os

from starfish.controller.tasks.abstract_r_task import AbstractRTask


class RMultipleImputation(AbstractRTask):
    """
    Federated Multiple Imputation implemented in R.

    Uses R's mice::mice() for MICE imputation, lm() for analysis,
    and mice::pool() for combining results via Rubin's rules.
    """

    def __init__(self, run):
        self.r_script_dir = os.path.join(
            os.path.dirname(__file__), 'scripts')
        super().__init__(run)
