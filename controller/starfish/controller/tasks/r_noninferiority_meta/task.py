import os

from starfish.controller.tasks.abstract_r_task import AbstractRTask


class RNoninferiorityMeta(AbstractRTask):
    """
    Federated site-stratified non-inferiority meta-analysis implemented in R.

    Each site returns a proportion difference and its standard error; the
    coordinating centre pools them by inverse-variance weighting using
    metafor's rma.uni.
    """

    def __init__(self, run):
        self.r_script_dir = os.path.join(
            os.path.dirname(__file__), 'scripts')
        super().__init__(run)
