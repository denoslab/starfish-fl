import json
from pathlib import Path

import numpy
import numpy as np
import pandas as pd

from starfish.controller.file.file_utils import gen_mid_artifacts_url, gen_all_mid_artifacts_url, gen_artifacts_url, \
    downloaded_artifacts_url
from starfish.controller.tasks.abstract_task import AbstractTask
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class LinearRegression(AbstractTask):

    def __init__(self, run):
        super().__init__(run)
        self.sample_size = None
        self.linearRegr = None  # Changed variable name
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
            
            # NOTE: We expect the data to be pre-processed before upload:
            # - One-hot encoding should be done before splitting the dataset
            # - Use preprocess_dataset.py script to create site1 and site2 CSV files
            # - The script can also be modified to split the dataset into more than two csv files
            # - This ensures all sites have identical feature sets (same columns)
            # Convert to DataFrame for easier preprocessing
            X_df = pd.DataFrame(X)
            
            # Log initial data info
            self.logger.debug(f'Original data shape: {X_df.shape}')
            self.logger.debug(f'Data types before conversion: {X_df.dtypes.value_counts().to_dict()}')
            
            # Convert columns to numeric where possible
            for col in X_df.columns:
                X_df[col] = pd.to_numeric(X_df[col], errors='ignore')
            
            self.logger.debug(f'Data types after conversion: {X_df.dtypes.value_counts().to_dict()}')
            
            # Separate numeric and categorical columns
            numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()
            
            self.logger.debug(f'Numeric columns: {len(numeric_cols)}, Categorical columns: {len(categorical_cols)}')
            
            # For federated learning, drop categorical columns to ensure consistency across sites
            if categorical_cols:
                self.logger.warning(f'Dropping {len(categorical_cols)} categorical columns for federated consistency')
                X_df = X_df[numeric_cols]
                self.logger.debug(f'After dropping categorical columns shape: {X_df.shape}')
            
            # Convert back to numpy array
            X = X_df.values.astype(float)
            
            # Ensure target variable is numeric and convert to numpy array of floats
            y = pd.to_numeric(pd.Series(y), errors='coerce')
            if y.isna().any():
                self.logger.error(f'Target variable contains non-numeric values that could not be converted')
                return False
            y = y.values.astype(float)
    
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

            # Initialize Linear regression model
            # LinearRegression doesn't have max_iter or warm_start
            # It uses closed-form solution (normal equation) by default
            self.linearRegr = sklearn.linear_model.LinearRegression()
            
            if not self.is_first_round():
                seq_no, round_no = self.get_previous_seq_and_round()
                directory = downloaded_artifacts_url(
                    self.run_id, seq_no, round_no)
                for path in Path(directory).rglob("*-{}-{}-artifacts".format(seq_no, round_no)):
                    with open(str(path), 'r') as f:
                        for line in f:
                            model = json.loads(line)
                            self.linearRegr.coef_ = np.asarray(
                                model['coef_'])
                            self.linearRegr.intercept_ = np.asarray(
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
        Linear regression uses closed-form solution (normal equation),
        so it computes optimal weights in one step.
        """
        self.logger.info('Starting training...')
        self.linearRegr.fit(self.X_train_scaled, self.y_train)
        score = self.linearRegr.score(self.X_test_scaled, self.y_test)
        self.logger.info(f'Training complete. Model R² score: {score}')
        to_upload = self.calculate_statistics()
        url = gen_mid_artifacts_url(
            self.run_id, self.cur_seq, self.get_round())
        self.logger.info("Upload: {} \n to: {}".format(to_upload, url))
        return self.save_artifacts(url, json.dumps(to_upload))

    def calculate_statistics(self):
        y_predict = self.linearRegr.predict(self.X_test_scaled)

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
            "coef_": self.linearRegr.coef_.tolist(),
            "intercept_": float(self.linearRegr.intercept_),  # intercept_ is scalar for linear regression
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
        coef = None
        intercept = None
        for mid_artifact_dict in download_mid_artifacts:
            sample = mid_artifact_dict['sample_size']
            self.sample_size = self.sample_size + sample
            weighted_coef = numpy.multiply(numpy.asarray(
                mid_artifact_dict['coef_']), sample)
            weighted_intercept = numpy.multiply(
                mid_artifact_dict['intercept_'], sample)
            if coef is None:
                coef = weighted_coef
            else:
                coef = numpy.add(coef, weighted_coef)
            if intercept is None:
                intercept = weighted_intercept
            else:
                intercept = numpy.add(intercept, weighted_intercept)

        if coef is not None and intercept is not None:
            self.linearRegr.coef_ = numpy.divide(coef, self.sample_size)
            self.linearRegr.intercept_ = numpy.divide(
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