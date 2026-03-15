import os
import pickle
from pathlib import Path

import numpy as np
import requests

from starfish.controller.file.file_utils import (
    create_if_not_exist,
    downloaded_artifacts_url,
    gen_all_mid_artifacts_url,
    gen_binary_artifacts_url,
    gen_binary_mid_artifacts_url,
    gen_logs_url,
    load_image_dataset_by_run,
    read_binary_file_from_url,
    read_file_from_url,
)
from starfish.controller.tasks.abstract_task import AbstractTask

router_url = os.getenv('ROUTER_URL')
router_username = os.getenv('ROUTER_USERNAME')
router_password = os.getenv('ROUTER_PASSWORD')


class FederatedUNet(AbstractTask):
    def __init__(self, run):
        super().__init__(run)
        self.model = None
        self.sample_size = 0
        self.train_images = None
        self.train_masks = None

    def _config(self):
        return self.tasks[self.cur_seq - 1].get('config', {})

    def _build_model(self):
        config = self._config()
        patch_size = int(config.get('patch_size', 64))
        architecture = str(config.get('architecture', 'resnet50')).lower()
        type_unet = str(config.get('type_Unet', 'unet')).lower()

        if type_unet != 'unet':
            raise ValueError("FederatedUNet currently supports type_Unet='unet' only")

        import segmentation_models as sm
        import tensorflow as tf
        from tensorflow import keras

        model = sm.Unet(
            architecture,
            classes=1,
            input_shape=(patch_size, patch_size, 1),
            encoder_weights=None,
            activation='sigmoid',
        )

        dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 0.5]))
        focal_loss = sm.losses.BinaryFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)

        model.compile(
            optimizer=keras.optimizers.Adam(float(config.get('learning_rate', 1e-4))),
            loss=total_loss,
            metrics=[sm.metrics.IOUScore(), sm.metrics.FScore()],
        )

        dummy = tf.zeros((1, patch_size, patch_size, 1))
        _ = model(dummy)
        return model

    def _set_runtime_env(self):
        os.environ['SM_FRAMEWORK'] = 'tf.keras'
        os.environ['TF_USE_LEGACY_KERAS'] = '1'

    def _find_previous_artifact_file(self):
        seq_no, round_no = self.get_previous_seq_and_round()
        if not seq_no or not round_no:
            return None

        directory = downloaded_artifacts_url(self.run_id, seq_no, round_no)
        if not directory:
            return None

        pattern = '*-{}-{}-artifacts'.format(seq_no, round_no)
        for path in sorted(Path(directory).rglob(pattern)):
            return path
        return None

    def validate(self) -> bool:
        if self.is_first_round():
            return True
        return self.download_artifact()

    def prepare_data(self) -> bool:
        try:
            self._set_runtime_env()
            config = self._config()
            patch_size = int(config.get('patch_size', 64))

            self.train_images, self.train_masks = load_image_dataset_by_run(self.run_id, patch_size)
            if self.train_images is None or self.train_masks is None:
                self.logger.error('Failed to load segmentation dataset zip')
                return False
            if len(self.train_images) == 0:
                self.logger.error('Loaded empty dataset')
                return False

            self.sample_size = len(self.train_images)
            self.model = self._build_model()

            if not self.is_first_round():
                prev_artifact = self._find_previous_artifact_file()
                if prev_artifact is None:
                    self.logger.error('Previous artifact not found')
                    return False
                with open(prev_artifact, 'rb') as f:
                    weights = pickle.load(f)
                self.model.set_weights(weights)

            self.logger.info('Prepared {} samples for local training'.format(self.sample_size))
            return True
        except Exception as e:
            self.logger.error('prepare_data failed due to {}'.format(e))
            return False

    def training(self) -> bool:
        try:
            config = self._config()
            local_epochs = int(config.get('local_epochs', 1))
            batch_size = int(config.get('batch_size', 8))

            train_x = np.expand_dims(self.train_images.astype(np.float32), axis=-1)
            train_y = np.expand_dims(self.train_masks.astype(np.float32), axis=-1)

            history = self.model.fit(
                train_x,
                train_y,
                batch_size=batch_size,
                epochs=local_epochs,
                verbose=0,
                shuffle=True,
            )

            metrics = {
                'loss': float(history.history.get('loss', [0.0])[-1]),
                'iou_score': float(history.history.get('iou_score', [0.0])[-1]),
                'f1-score': float(history.history.get('f1-score', [0.0])[-1]),
            }

            payload = {
                'weights': self.model.get_weights(),
                'n_samples': self.sample_size,
                'metrics': metrics,
            }
            url = gen_binary_mid_artifacts_url(self.run_id, self.cur_seq, self.get_round())
            return self.save_artifacts(url, payload)
        except Exception as e:
            self.logger.error('training failed due to {}'.format(e))
            return False

    def do_aggregate(self) -> bool:
        try:
            directory = gen_all_mid_artifacts_url(self.project_id, self.batch_id)
            if not directory:
                return False

            pattern = '*-{}-{}-mid-artifacts'.format(self.cur_seq, self.get_round())
            participant_results = []
            for path in sorted(Path(directory).rglob(pattern)):
                with open(path, 'rb') as f:
                    participant_results.append(pickle.load(f))

            if not participant_results:
                self.logger.warning('No participant mid-artifacts found for aggregation')
                return False

            total_samples = sum(int(item.get('n_samples', 0)) for item in participant_results)
            if total_samples <= 0:
                self.logger.warning('Invalid total sample size for aggregation')
                return False

            agg_weights = None
            for result in participant_results:
                n_samples = int(result.get('n_samples', 0))
                if n_samples <= 0:
                    continue
                frac = n_samples / total_samples
                weights = result['weights']
                if agg_weights is None:
                    agg_weights = [np.zeros_like(w) for w in weights]
                for idx in range(len(agg_weights)):
                    agg_weights[idx] += frac * weights[idx]

            if agg_weights is None:
                return False

            artifact_url = gen_binary_artifacts_url(self.run_id, self.cur_seq, self.get_round())
            if not self.save_artifacts(artifact_url, agg_weights):
                return False

            return self.upload(True)
        except Exception as e:
            self.logger.error('do_aggregate failed due to {}'.format(e))
            return False

    def upload(self, is_artifact: bool) -> bool:
        task_round = self.get_round()
        if not task_round:
            return False

        files_data = {}
        data = {
            'run': self.run_id,
            'task_seq': self.cur_seq,
            'round_seq': task_round,
        }

        if is_artifact:
            artifact_url = gen_binary_artifacts_url(self.run_id, self.cur_seq, task_round)
            if artifact_url:
                files_data['artifacts'] = read_binary_file_from_url(artifact_url)
        else:
            mid_url = gen_binary_mid_artifacts_url(self.run_id, self.cur_seq, task_round)
            logs_url = gen_logs_url(self.run_id, self.cur_seq, task_round)
            if mid_url:
                files_data['mid_artifacts'] = read_binary_file_from_url(mid_url)
            if logs_url:
                files_data['logs'] = read_file_from_url(logs_url)

        if not any(files_data.values()):
            self.logger.debug('Files data is empty. Ignore upload')
            return True

        try:
            response = requests.post(
                '{}/runs-action/upload/'.format(router_url),
                auth=(router_username, router_password),
                data=data,
                files=files_data,
            )
            return response.status_code == 200
        finally:
            for f in files_data.values():
                try:
                    if f:
                        f.close()
                except Exception:
                    pass

    def save_artifacts(self, url, content):
        if content is None:
            return False
        create_if_not_exist(url)
        try:
            with open(url, 'wb') as f:
                pickle.dump(content, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e:
            self.logger.error('Error while saving binary artifact due to {}'.format(e))
            return False
