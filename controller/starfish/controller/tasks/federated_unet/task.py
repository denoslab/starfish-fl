"""Federated UNet image segmentation task.

Implements federated learning for binary image segmentation using a UNet
architecture with FedAvg (Federated Averaging) aggregation. Each site
trains locally on its own images and only model weights are exchanged.

Requires the ``unet`` dependency group (TensorFlow, segmentation-models).
"""

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
    """
    Federated image segmentation using UNet with FedAvg aggregation.

    Each site uploads a zip containing ``images/`` and ``masks/`` directories.
    Images are converted to grayscale and resized to ``patch_size x patch_size``.
    Masks are binarized (threshold 127) for binary segmentation.

    Aggregation uses sample-weighted averaging of model weights (FedAvg).

    Config Parameters
    -----------------
    patch_size : int, default 64
        Input image dimension (images resized to patch_size x patch_size).
    architecture : str, default 'resnet50'
        Encoder backbone for the UNet (e.g., 'resnet50', 'mobilenetv2').
    type_Unet : str, default 'unet'
        Segmentation architecture type. Only 'unet' is currently supported.
    local_epochs : int, default 1
        Number of local training epochs per FL round.
    batch_size : int, default 8
        Training mini-batch size.
    learning_rate : float, default 1e-4
        Adam optimizer learning rate.
    """

    def __init__(self, run):
        super().__init__(run)
        self.model = None
        self.sample_size = 0
        self.train_images = None
        self.train_masks = None

    def _config(self):
        """Return the config dict for the current task sequence."""
        return self.tasks[self.cur_seq - 1].get('config', {})

    def _build_model(self):
        """
        Build and compile a UNet segmentation model.

        Uses the ``segmentation-models`` library with a configurable encoder
        backbone. The loss function combines Dice loss and Binary Focal loss
        for robust training on imbalanced segmentation masks.

        Returns
        -------
        tensorflow.keras.Model
            Compiled UNet model ready for training.
        """
        config = self._config()
        patch_size = int(config.get('patch_size', 64))
        architecture = str(config.get('architecture', 'resnet50')).lower()
        type_unet = str(config.get('type_Unet', 'unet')).lower()

        if type_unet != 'unet':
            raise ValueError("FederatedUNet currently supports type_Unet='unet' only")

        # Lazy imports: TensorFlow is heavy and only available in the unet group
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

        # Combined loss: Dice handles class imbalance, Focal penalizes
        # hard-to-classify pixels
        dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 0.5]))
        focal_loss = sm.losses.BinaryFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)

        model.compile(
            optimizer=keras.optimizers.Adam(float(config.get('learning_rate', 1e-4))),
            loss=total_loss,
            metrics=[sm.metrics.IOUScore(), sm.metrics.FScore()],
        )

        # Trigger weight initialization by running a dummy forward pass
        dummy = tf.zeros((1, patch_size, patch_size, 1))
        _ = model(dummy)
        return model

    def _set_runtime_env(self):
        """Configure environment variables required by segmentation-models."""
        os.environ['SM_FRAMEWORK'] = 'tf.keras'
        os.environ['TF_USE_LEGACY_KERAS'] = '1'

    def _find_previous_artifact_file(self):
        """
        Locate the aggregated weights artifact from the previous round.

        Returns
        -------
        Path or None
            Path to the previous round's artifact file, or None if not found.
        """
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
        """Validate readiness: first round always passes; later rounds need prior artifacts."""
        if self.is_first_round():
            return True
        return self.download_artifact()

    def prepare_data(self) -> bool:
        """
        Load the image dataset and build (or restore) the UNet model.

        Reads image/mask pairs from the uploaded zip, builds the model, and
        restores weights from the previous round's aggregated artifact when
        resuming a multi-round run.
        """
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

            # Restore aggregated weights from the previous FL round
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
        """
        Run local training and save mid-artifacts.

        Trains the UNet on this site's images for ``local_epochs`` epochs,
        then serializes the model weights, sample count, and training metrics
        (loss, IoU, F1) as a pickle mid-artifact for aggregation.
        """
        try:
            config = self._config()
            local_epochs = int(config.get('local_epochs', 1))
            batch_size = int(config.get('batch_size', 8))

            # Add channel dimension for grayscale: (N, H, W) -> (N, H, W, 1)
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

            # Package weights + metadata for the coordinator to aggregate
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
        """
        Aggregate model weights from all participants using FedAvg.

        Computes a sample-weighted average of each participant's model weights,
        where the weight for each site is proportional to its local sample count.
        The aggregated weights are saved and uploaded to the router.
        """
        try:
            directory = gen_all_mid_artifacts_url(self.project_id, self.batch_id)
            if not directory:
                return False

            # Collect mid-artifacts from every participant
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

            # FedAvg: weighted average of model weights by sample count
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
        """
        Upload artifacts or mid-artifacts to the router.

        Parameters
        ----------
        is_artifact : bool
            If True, upload the final aggregated artifact.
            If False, upload mid-artifacts and logs from local training.

        Returns
        -------
        bool
            True if upload succeeded or there was nothing to upload.
        """
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
        """
        Serialize and save artifact content as a pickle file.

        Parameters
        ----------
        url : str
            Local filesystem path for the artifact.
        content : object
            Python object to pickle (model weights or payload dict).

        Returns
        -------
        bool
            True if saved successfully.
        """
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
