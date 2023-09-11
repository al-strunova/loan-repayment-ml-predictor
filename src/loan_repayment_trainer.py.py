import logging

import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, early_stopping
from src.features_transformer import FeaturesTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Setting up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class TrainLoanRepaymentModel:
    """Class to train and save a loan repayment prediction model."""

    def __init__(self, train_file_path: str, client_profile_data_path: str = None,
                 history_data_path: str = None, bki_data_path: str = None,
                 payments_data_path: str = None):
        """Constructor to initialize paths and load training data.

        Args:
        - train_file_path: Path to the main training data.
        - client_profile_data_path: Path to the client profile data.
        - history_data_path: Path to the applications history data.
        - bki_data_path: Path to the bki data.
        - payments_data_path: Path to the payment's data.
        """
        # Paths for data sources
        self.train_data = pd.read_csv(train_file_path)
        logging.info(f"Loaded training data from {train_file_path} with {self.train_data.shape[0]} "
                     f"rows and {self.train_data.shape[1]} columns.")
        self.client_profile_data_path = client_profile_data_path
        self.history_data_path = history_data_path
        self.bki_data_path = bki_data_path
        self.payments_data_path = payments_data_path

        # Features and target variable extraction
        self.X = self.train_data.drop(columns=["TARGET"])
        self.y = self.train_data["TARGET"]

        # Data split attributes
        self.X_valid = None
        self.X_train = None
        self._prepare_data_split()  # Split data into train and validation

        # Model and preprocessing attributes
        self.model = None
        self.preprocessor = None

    def _prepare_data_split(self):
        """Splits the data into train, validation, and test sets."""
        logging.info("Splitting data into train, validation, and test sets.")
        X_train, X_valid, y_train, y_valid = train_test_split(
            self.X, self.y, test_size=0.1, random_state=1234, stratify=self.y
        )
        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(
            X_valid, y_valid, test_size=0.2, random_state=1234, stratify=y_valid
        )
        self.X_train, self.y_train = X_train, y_train

    def _get_preprocessing_pipeline(self):
        """Creates and returns the preprocessing pipeline."""
        logging.info("Setting up preprocessing pipeline.")
        return make_pipeline(
            FeaturesTransformer(self.client_profile_data_path, self.history_data_path,
                                self.bki_data_path, self.payments_data_path),
            make_column_transformer(
                (make_pipeline(SimpleImputer(strategy='constant', fill_value='Missing'),
                               OneHotEncoder(sparse_output=False)),
                 make_column_selector(dtype_exclude='number')),
                (SimpleImputer(strategy='median'), make_column_selector(dtype_include='number')),
                remainder='passthrough'
            )
        )

    def preprocess_data(self, X, fit=False):
        """Preprocesses the data using the preprocessing pipeline.

        Args:
        - X: The input data.
        - fit: Whether to fit the preprocessing pipeline or just transform.

        Returns:
        - Preprocessed data.
        """
        logging.info("Preprocessing data.")
        if fit:
            logging.info("Fitting the preprocessor.")
            self.preprocessor = self._get_preprocessing_pipeline()
            return self.preprocessor.fit_transform(X)
        else:
            if self.preprocessor is None:
                raise ValueError(
                    "Preprocessor has not been fitted. Make sure to preprocess training data with fit=True first.")
            return self.preprocessor.transform(X)

    def save_preprocessing_pipeline(self, path='../models/preprocessing_pipeline.pkl'):
        """Saves the preprocessing pipeline to disk.

        Args:
        - path: Path to save the preprocessing pipeline.
        """
        logging.info(f"Saving preprocessing pipeline to {path}.")
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been initialized. Make sure to preprocess training data first.")
        joblib.dump(self.preprocessor, path)

    def save_model(self, path='../models/model.pkl'):
        """Saves the trained model to disk.

        Args:
        - path: Path to save the model.
        """
        logging.info(f"Saving model to {path}.")
        joblib.dump(self.model, path)

    def train_model(self):
        """Trains the LGBMClassifier model."""
        logging.info("Starting model training.")
        lgbm_params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "learning_rate": 0.01,
            "n_estimators": 1139,
            "max_depth": 8,
            "num_leaves": 10000,
            "bagging_fraction": 0.3,
            "min_data_in_leaf": 1000,
            "silent": -1,
            "verbose": -1
        }
        model = LGBMClassifier(**lgbm_params)

        # Preprocess the training and validation data
        self.X_train = self.preprocess_data(self.X_train, fit=True)
        self.X_valid = self.preprocess_data(self.X_valid)

        # Early stopping for LightGBM
        stopper = early_stopping(stopping_rounds=100, first_metric_only=False)

        # Train the model
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_valid, self.y_valid)],
            eval_metric="auc",
            callbacks=[stopper]
        )
        # Store the trained model
        self.model = model
        logging.info("Model training completed.")
        self.predict_and_scores_roc_auc()

    def predict_and_scores_roc_auc(self):

        # Make a prediction on different validation train datasets
        pred_train = self.model.predict_proba(self.X_train)
        pred_valid = self.model.predict_proba(self.X_valid)
        self.X_test = self.preprocess_data(self.X_test)
        pred_test = self.model.predict_proba(self.X_test)

        # Score roc_auc
        train_score = round(roc_auc_score(self.y_train, pred_train[:, 1]), 3)
        valid_score = round(roc_auc_score(self.y_valid, pred_valid[:, 1]), 3)
        test_score = round(roc_auc_score(self.y_test, pred_test[:, 1]), 3)

        logging.info(f'Train-score: {train_score}, Valid-score: {valid_score}, Test-score: {test_score}, ')
        return train_score, valid_score, test_score


if __name__ == "__main__":
    # Train and save the model and preprocessing pipeline
    trainer = TrainLoanRepaymentModel(
        train_file_path='../data/train.csv',
        client_profile_data_path='../data/client_profile.csv',
        history_data_path='../data/applications_history.csv',
        bki_data_path='../data/bki.csv',
        payments_data_path='../data/payments.csv'
    )
    logging.info("Initiating model training.")
    trainer.train_model()

    logging.info("Saving preprocessing pipeline.")
    trainer.save_preprocessing_pipeline()

    logging.info("Saving trained model.")
    trainer.save_model()
    print("DONE")
