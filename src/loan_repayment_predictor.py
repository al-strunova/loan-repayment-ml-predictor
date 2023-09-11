import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class PredictLoanRepaymentModel:
    """
    A class for making loan repayment predictions using a pre-trained model and preprocessing pipeline.

    Parameters:
    - model_location: str, default='model.pkl'
      The file location of the pre-trained machine learning model.

    - preprocessing_pipeline_location: str, default='preprocessing_pipeline.pkl'
      The file location of the saved preprocessing pipeline.
    """
    # Determine the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct absolute paths to the model and preprocessing pipeline files
    model_path = os.path.join(script_directory, '..', 'models', 'model.pkl')
    preprocessing_pipeline_path = os.path.join(script_directory, '..', 'models', 'preprocessing_pipeline.pkl')

    def __init__(self,
                 model_absolute_path=model_path,
                 preprocessing_pipeline_absolute_path=preprocessing_pipeline_path):
        """
        Initialize the PredictLoanRepaymentModel.

        Parameters:
        - model_location: str, default='model.pkl'
          The file location of the pre-trained machine learning model.

        - preprocessing_pipeline_location: str, default='preprocessing_pipeline.pkl'
          The file location of the saved preprocessing pipeline.
        """

        try:
            logging.info("Leading Model and preprocessing pipeline...")
            self.model = joblib.load(model_absolute_path)
            self.preprocessing = joblib.load(preprocessing_pipeline_absolute_path)

            logging.info("Model and preprocessing pipeline loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"File not found error: {e}")
        except Exception as e:
            logging.error(f"An error occurred while loading the model and preprocessing pipeline: {e}")

    def predict(self, X_new):
        """
        Predict loan repayment probabilities for new data.

        Parameters:
        - X_new: pandas.DataFrame or numpy.ndarray
          The new data to make predictions on.

        Returns:
        - predictions: pandas.DataFrame
          A DataFrame containing predicted probabilities (loan repayment probabilities)
          along with the corresponding APPLICATION_NUMBER.
        """

        test_ids = X_new['APPLICATION_NUMBER'].copy()

        # Step 1: Transform the input data using the preprocessing pipeline
        logging.info("Starting prediction process...")
        X_transformed = self.preprocessing.transform(X_new)

        # Step 2: Use the loaded model to make predictions
        logging.info("Making predictions using the loaded model...")
        predict = np.round(self.model.predict_proba(X_transformed)[:, 1], 4)
        results = pd.DataFrame({
            "APPLICATION_NUMBER": test_ids,
            "TARGET": predict
        })

        logging.info("Prediction process completed.")

        # Step 3: Return the predictions
        return results


if __name__ == "__main__":
    # Create an instance of the model
    model = PredictLoanRepaymentModel()

    # Make predictions
    logging.info("Loading test data...")
    test_data = pd.read_csv('../data/test.csv')

    logging.info("Making predictions on test data...")
    predictions = model.predict(test_data)

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fn = f"../predictions/loan_repayment_predictions_{timestamp}.csv"
    logging.info(f"Result successfully saved to {fn}")
    predictions.to_csv(fn, index=False)
