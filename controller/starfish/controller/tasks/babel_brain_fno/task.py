"""BabelBrainFno: federated fine-tuning of tFUS-FNO on local BabelBrain samples.

SF-03 scope: the task reads the site's own sample store, see ``store.py``,
and needs no dataset upload. Training and aggregation arrive with SF-04;
until then ``training`` and ``do_aggregate`` fail the round on purpose.
"""

import os

from starfish.controller.file import file_utils
from starfish.controller.tasks.abstract_task import AbstractTask
from starfish.controller.tasks.babel_brain_fno.store import SampleStore, StoreError
from starfish.controller.tasks.data_source import BABELBRAIN_STORE, validate_data_source

DEFAULT_MIN_SAMPLES = 20
HASH_CACHE_NAME = os.path.join('babelbrain_fl', 'sha256_cache.json')


class BabelBrainFno(AbstractTask):
    """
    No LLM calls in the BabelBrain path: agent hooks are never loaded, even
    if the task config asks for them.

    Config Parameters
    -----------------
    data_source : dict, required
        ``{"type": "babelbrain_store", "bucket_hz": 250000}``. The store
        location comes from the site's ``BABELBRAIN_FL_STORE``.
    min_samples : int, default 20
        Refuse to train with fewer local train samples than this.
    """

    agents_allowed = False

    def __init__(self, run):
        super().__init__(run)
        self.train_records = []
        self.val_records = []

    def _config(self):
        return self.tasks[self.cur_seq - 1].get('config', {})

    def validate(self) -> bool:
        if self.is_first_round():
            return True
        self.logger.error(
            'BabelBrainFno multi-round training is not implemented yet, see SF-04')
        return False

    def open_store(self):
        """Open this site's store. The location never comes from task config."""
        cache_path = os.path.join(file_utils.base_folder, HASH_CACHE_NAME)
        return SampleStore.from_env(cache_path=cache_path, logger=self.logger)

    def prepare_data(self) -> bool:
        config = self._config()
        data_source = config.get('data_source')
        error = validate_data_source(data_source)
        if error is None and data_source.get('type') != BABELBRAIN_STORE:
            error = 'BabelBrainFno needs a {} data_source'.format(
                BABELBRAIN_STORE)
        if error:
            self.logger.error(error)
            return False
        bucket_hz = data_source['bucket_hz']
        min_samples = int(config.get('min_samples', DEFAULT_MIN_SAMPLES))

        try:
            store = self.open_store()
            self.train_records = store.samples(bucket_hz, 'train')
            self.val_records = store.samples(bucket_hz, 'val')
        except StoreError as e:
            self.logger.error('Sample store unavailable: {}'.format(e))
            return False

        self.logger.info('Sample store at {} Hz: {} train, {} val'.format(
            bucket_hz, len(self.train_records), len(self.val_records)))
        if len(self.train_records) < min_samples:
            self.logger.error('Only {} train samples, fewer than min_samples {}'.format(
                len(self.train_records), min_samples))
            return False
        return True

    def training(self) -> bool:
        self.logger.error(
            'BabelBrainFno training is not implemented yet, see SF-04')
        return False

    def do_aggregate(self) -> bool:
        self.logger.error(
            'BabelBrainFno aggregation is not implemented yet, see SF-04')
        return False
