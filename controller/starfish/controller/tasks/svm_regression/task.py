import json
from pathlib import Path

import numpy
import numpy as np

from starfish.controller.file.file_utils import gen_mid_artifacts_url, gen_all_mid_artifacts_url, gen_artifacts_url, \
    downloaded_artifacts_url
from starfish.controller.tasks.abstract_task import AbstractTask
import sklearn.svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# NOTE: Implementation is not completed yet.

class SvmRegression(AbstractTask):   # Also called Support Vector Regression (SVR)

    def __init__(self, run):
        super().__init__(run)
        self.sample_size = None
        self.svmRegr = None
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

            # Initialize SVM regression model
            self.svmRegr = sklearn.svm.SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1,
            )
            if not self.is_first_round():
                seq_no, round_no = self.get_previous_seq_and_round()
                directory = downloaded_artifacts_url(
                    self.run_id, seq_no, round_no)
                for path in Path(directory).rglob("*-{}-{}-artifacts".format(seq_no, round_no)):
                    with open(str(path), 'r') as f:
                        for line in f:
                            model = json.loads(line)
                            self.svmRegr.support_vectors_ = np.asarray(
                                model['support_vectors_'])
                            self.svmRegr.dual_coef_ = np.asarray(
                                model['dual_coef_'])
                            self.svmRegr.intercept_ = np.asarray(
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
        SVM regression uses support vectors to define the regression function.
        """
        self.logger.info('Starting training...')
        self.svmRegr.fit(self.X_train_scaled, self.y_train)
        score = self.svmRegr.score(self.X_test_scaled, self.y_test)
        self.logger.info(f'Training complete. Model R² score: {score}')
        to_upload = self.calculate_statistics()
        url = gen_mid_artifacts_url(
            self.run_id, self.cur_seq, self.get_round())
        self.logger.info("Upload: {} \n to: {}".format(to_upload, url))
        return self.save_artifacts(url, json.dumps(to_upload))

    def calculate_statistics(self):
        y_predict = self.svmRegr.predict(self.X_test_scaled)

        # Regression metrics
        mse = mean_squared_error(self.y_test, y_predict)
        self.logger.info(f'Mean Squared Error: {mse}')
        
        rmse = np.sqrt(mse)
        self.logger.info(f'Root Mean Squared Error: {rmse}')
        
        mae = mean_absolute_error(self.y_test, y_predict)
        self.logger.info(f'Mean Absolute Error: {mae}')
        
        r2 = r2_score(self.y_test, y_predict)
        self.logger.info(f'R² Score: {r2}')

        return {
            "sample_size": self.sample_size,
            "dual_coef": self.svmRegr.dual_coef_.tolist(),
            "intercept": float(self.svmRegr.intercept_[0]),
            "metric_mse": mse,
            "metric_rmse": rmse,
            "metric_mae": mae,
            "metric_r2": r2
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
        dual_coef = None
        intercept = None
        
        for mid_artifact_dict in download_mid_artifacts:
            sample = mid_artifact_dict['sample_size']
            self.sample_size = self.sample_size + sample
            
            weighted_dual_coef = numpy.multiply(numpy.asarray(
                mid_artifact_dict['dual_coef']), sample)
            weighted_intercept = numpy.multiply(
                mid_artifact_dict['intercept'], sample)
            
            if dual_coef is None:
                dual_coef = weighted_dual_coef
            else:
                dual_coef = numpy.add(dual_coef, weighted_dual_coef)
            if intercept is None:
                intercept = weighted_intercept
            else:
                intercept = numpy.add(intercept, weighted_intercept)

        if dual_coef is not None and intercept is not None:
            self.svmRegr.dual_coef_ = numpy.divide(dual_coef, self.sample_size)
            self.svmRegr.intercept_ = numpy.array([numpy.divide(intercept, self.sample_size)])
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
                "Not able to calculate dual coefficients and intercept due to invalid mid artifact")
            return False