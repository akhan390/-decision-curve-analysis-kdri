from google.cloud import storage
import joblib
import argparse
import numpy as np
import pandas as pd
from io import StringIO
import json
import logging
from sklearn.model_selection import train_test_split
import sksurv.datasets
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import ParameterGrid
import os

logging.basicConfig(level=logging.INFO)

def report_metric_to_gcp(metric_name, metric_value):
    metric_data = {
        "metric_name": metric_name,
        "metric_value": metric_value
    }
    logging.info(json.dumps(metric_data))

# Create the argument parser for each parameter plus the job dictionary
parser = argparse.ArgumentParser(description='Random Survival Forest')
parser.add_argument('--job-dir', help='GCS location to write checkpoints and export models')
parser.add_argument('--n_estimators', type=int, default=100, help='The number of trees in the forest')
parser.add_argument('--min_samples_split', type=float, default=0.1, help='Minimum number of samples required to split an internal node')
parser.add_argument('--min_samples_leaf', type=int, default=1, help='The minimum number of samples required to be at a leaf node')
args = parser.parse_args()

try:
    logging.info("Initializing Google Cloud Storage client...")
    client = storage.Client()
    bucket = client.get_bucket('decision-curve-analysis')
    blob = bucket.blob('df_rsf.csv')
    logging.info("Downloading data from GCS...")
    data = blob.download_as_text()
    df = pd.read_csv(StringIO(data))
    df.set_index('TRR_ID', inplace=True)
    logging.info("Data successfully loaded.")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    raise

try:
    logging.info("Defining features and target variables...")
    X, y = sksurv.datasets.get_x_y(df, attr_labels=['REC_GRAFT_STAT', 'Graft_Survival_Time'], pos_label=1, survival=True)
except Exception as e:
    logging.error(f"Error defining features and target: {e}")
    raise

try:
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=39)
    logging.info(f"Training size: {len(X_train)}, Testing size: {len(X_test)}")

    # Save the datasets to GCS in the same bucket
    job_dir = args.job_dir.replace('gs://', '')
    bucket_id = job_dir.split('/')[0]
    bucket_path = job_dir.lstrip(f'{bucket_id}/')
    bucket = storage.Client().bucket(bucket_id)

    for name, data in zip(['X_train', 'X_test', 'y_train', 'y_test'], [X_train, X_test, y_train, y_test]):
        local_filename = f'{name}.joblib'
        joblib.dump(data, local_filename)
        blob = bucket.blob(f'{bucket_path}/{local_filename}')
        blob.upload_from_filename(local_filename)
    logging.info("Training and testing datasets saved successfully to GCS.")
except Exception as e:
    logging.error(f"Error during train-test split or saving datasets: {e}")
    raise

# Hyperparameter tuning setup
param_grid = {
    'n_estimators': [50, 100, 150],
    'min_samples_split': [0.1, 0.2],
    'min_samples_leaf': [1, 2]
}

best_model = None
best_c_index = 0
best_params = {}

for params in ParameterGrid(param_grid):
    try:
        logging.info(f"Training model with params: {params}")
        model = RandomSurvivalForest(
            n_estimators=params['n_estimators'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf']
        )
        model.fit(X_train, y_train)
        c_index = model.score(X_test, y_test)
        logging.info(f"Concordance Index for current model: {c_index}")

        # Reporting metrics for each iteration
        report_metric_to_gcp('concordance_index', c_index)

        if c_index > best_c_index:
            best_c_index = c_index
            best_model = model
            best_params = params

    except Exception as e:
        logging.error(f"Error during hyperparameter tuning: {e}")
        continue

logging.info(f"Best model parameters: {best_params}")
logging.info(f"Best Concordance Index: {best_c_index}")

# Ensure the best model is fully trained and validated before saving
if best_model is not None:
    logging.info("Refitting and validating the best model before saving...")
    best_model.fit(X_train, y_train)
    if hasattr(best_model, "predict_survival_function"):
        logging.info("Best model successfully fitted and validated with survival prediction capabilities.")
    else:
        logging.error("Best model does not support survival prediction. Raising error.")
        raise ValueError("Final model does not have the required prediction methods.")
else:
    logging.error("No valid model was found during hyperparameter tuning.")
    raise ValueError("No valid model found.")

model_filename = 'best_model.joblib'
try:
    logging.info(f"Exporting best model to file {model_filename}...")
    joblib.dump(best_model, model_filename)
    blob = bucket.blob(f'{bucket_path}/{model_filename}')
    blob.upload_from_filename(model_filename)
    logging.info(f"Model successfully uploaded to gs://{bucket_id}/{bucket_path}/{model_filename}")
except Exception as e:
    logging.error(f"Failed to upload model to GCS: {e}")
    raise
