import json
from pathlib import Path

import numpy
import numpy as np

from starfish.controller.file.file_utils import gen_mid_artifacts_url, gen_all_mid_artifacts_url, gen_artifacts_url, \
    downloaded_artifacts_url
from starfish.controller.tasks.abstract_task import AbstractTask
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings

warnings.filterwarnings('ignore')


class LogisticRegression(AbstractTask):

    def __init__(self, run):
        super().__init__(run)
        self.sample_size = None
        self.logisticRegr = None
        self.X_train_scaled = None
        self.y_train = None
        self.X_test_scaled = None
        self.y_test = None

    def prepare_data(self) -> bool:
        # load dataset
        self.logger.debug('Loading dataset for run {} ...'.format(self.run_id))
        X, y = self.read_dataset(self.run_id)
        if X is not None and len(X) > 0 and y is not None and len(y) > 0:
            self.sample_size = len(y)
            # Split the data into training and testing sets
            X_train, X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            # Standardize the numerical features
            scaler = StandardScaler()
            self.X_train_scaled = scaler.fit_transform(X_train)
            self.X_test_scaled = scaler.transform(X_test)
            self.logger.debug(
                f'Training data shape: {self.X_train_scaled.shape}')
            self.logger.debug(f'Training label shape: {self.y_train.shape}')
            self.logger.debug(f'Test data shape: {self.X_test_scaled.shape}')
            self.logger.debug(f'Test label shape: {self.y_test.shape}')

            # Initialize Logistic regression model
            self.logisticRegr = sklearn.linear_model.LogisticRegression(
                penalty="l2",
                max_iter=1,  # local epoch
                warm_start=True,  # prevent refreshing weights when fitting
            )
            if not self.is_first_round():
                seq_no, round_no = self.get_previous_seq_and_round()
                directory = downloaded_artifacts_url(
                    self.run_id, seq_no, round_no)
                for path in Path(directory).rglob("*-{}-{}-artifacts".format(seq_no, round_no)):
                    with open(str(path), 'r') as f:
                        for line in f:
                            model = json.loads(line)
                            self.logisticRegr.coef_ = np.asarray(
                                model['coef_'])
                            self.logisticRegr.intercept_ = np.asarray(
                                model['intercept_'])
            return True
        else:
            self.logger.warning("Data set is not ready")
        return False

    def validate(self) -> bool:
        """
        This step is used to load and validate the input data.
        """
        task_round = self.get_round()
        validate_log = "Run {} - task -{} - round {} task begins".format(
            self.run_id, self.cur_seq, task_round)
        self.logger.debug(validate_log)
        return self.download_artifact()

    def training(self) -> bool:
        """
        This step is used for training.
        all parameters not specified are set to their defaults
        default solver is incredibly slow thats why we change it
        """
        self.logger.info('Starting training...')
        self.logisticRegr.fit(self.X_train_scaled, self.y_train)
        score = self.logisticRegr.score(self.X_test_scaled, self.y_test)
        self.logger.info(f'Training complete. Model score: {score}')
        to_upload = self.calculate_statistics()
        url = gen_mid_artifacts_url(
            self.run_id, self.cur_seq, self.get_round())
        self.logger.info("Upload: {} \n to: {}".format(to_upload, url))
        return self.save_artifacts(url, json.dumps(to_upload))

    def calculate_statistics(self):
        y_predict = self.logisticRegr.predict(self.X_test_scaled)

        # Accuracy metric
        accuracy = accuracy_score(self.y_test, y_predict)
        self.logger.info(f'Accuracy: {accuracy}')
        report = classification_report(self.y_test, y_predict)
        self.logger.info(f'Classification report : \n  {report}')

        # Documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
        # ROC-AUC
        auc = roc_auc_score(self.y_test, y_predict)
        self.logger.info(f'AUC: {auc}')

        # Use confusion matrix to calculate the metrics
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_predict).ravel()

        accuracy = (tp + tn) / (tn + fp + fn + tp)
        self.logger.info(f'Accuracy: {accuracy}')

        sensitivity = tp / (tp + fn)
        self.logger.info(f'Sensitivity: {sensitivity}')

        specificity = tn / (tn + fp)
        self.logger.info(f'Specificity: {specificity}')

        npv = tn / (tn + fn)
        self.logger.info(f'NPV: {npv}')

        ppv = tp / (tp + fp)
        self.logger.info(f'PPV: {ppv}')

        return {
            "sample_size": self.sample_size,
            "coef_": self.logisticRegr.coef_.tolist(),
            "intercept_": self.logisticRegr.intercept_.tolist(),
            "metric_acc": accuracy,
            "metric_auc": auc,
            "metric_sensitivity": sensitivity,
            "metric_specificity": specificity,
            "metric_npv": npv,
            "metric_ppv": ppv
        }

    def do_aggregate(self) -> bool:
        download_mid_artifacts = []
        directory = gen_all_mid_artifacts_url(self.project_id, self.batch_id)
        for path in Path(directory).rglob("*-{}-{}-mid-artifacts".format(self.cur_seq, self.get_round())):
            with open(str(path), 'r') as f:
                for line in f:
                    download_mid_artifacts.append(json.loads(line))

        self.logger.debug(
            "Download mid artifacts: {}".format(download_mid_artifacts))

        self.sample_size = 0
        coef = None
        intercept = None
        for mid_artifact_dict in download_mid_artifacts:
            sample = mid_artifact_dict['sample_size']
            self.sample_size = self.sample_size + sample
            weighted_coef = numpy.multiply(numpy.asarray(
                mid_artifact_dict['coef_']), sample)
            weighted_intercept = numpy.multiply(numpy.asarray(
                mid_artifact_dict['intercept_']), sample)
            if coef is None:
                coef = weighted_coef
            else:
                coef = numpy.add(coef, weighted_coef)
            if intercept is None:
                intercept = weighted_intercept
            else:
                intercept = numpy.add(intercept, weighted_intercept)

        if coef is not None and intercept is not None:
            self.logisticRegr.coef_ = numpy.divide(coef, self.sample_size)
            self.logisticRegr.intercept_ = numpy.divide(
                intercept, self.sample_size)
            to_upload = self.calculate_statistics()
            url = gen_artifacts_url(
                self.run_id, self.cur_seq, self.get_round())
            self.logger.info("Upload: {} \n to: {}".format(to_upload, url))
            if self.save_artifacts(url, json.dumps(to_upload)):
                self.upload(True)
                return True
            else:
                return False
        else:
            self.logger.warning(
                "Not able to calculate coef and intercept due to invalid mid artifact")
            return False
