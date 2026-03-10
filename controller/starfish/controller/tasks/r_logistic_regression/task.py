import os

from starfish.controller.tasks.abstract_r_task import AbstractRTask


class RLogisticRegression(AbstractRTask):
    """
    Federated logistic regression implemented in R.

    Uses R's glm(family=binomial) for local training and weighted
    coefficient averaging for aggregation.
    """

    def __init__(self, run):
        self.r_script_dir = os.path.join(
            os.path.dirname(__file__), 'scripts')
        super().__init__(run)
