import os
import pandas as pd
import joblib
from typing import Tuple
from sklearn.model_selection import train_test_split
from google.cloud import storage
from model import DelayModel

# Configuration
DATA_PATH = "data/data.csv"
MODEL_OUTPUT = "model.joblib"
BUCKET_NAME = os.getenv("MODEL_BUCKET", "bucket-latam-challenge")

def load_and_prep_data(path: str) -> pd.DataFrame:
    """Loads raw CSV data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found at {path}")
    return pd.read_csv(path)

def run_training_pipeline() -> None:
    """
    Executes the full training pipeline:
    1. Load data
    2. Preprocess using DelayModel logic
    3. Split data (using notebook random_state=42)
    4. Fit the model
    5. Serialize and Upload to GCS
    """
    print("Starting training pipeline...")
    
    # 1. Load Data
    raw_data = load_and_prep_data(DATA_PATH)
    
    # 2. Initialize Model Wrapper
    model_wrapper = DelayModel()
    
    # 3. Preprocess
    features, target = model_wrapper.preprocess(data=raw_data, target_column="delay")
    
    # 4. Split Data
    x_train, x_test, y_train, y_test = train_test_split(
        features, 
        target, 
        test_size=0.33, 
        random_state=42
    )
    
    print(f"Dataset split: Train={x_train.shape[0]} | Test={x_test.shape[0]}")

    # 5. Fit Model
    print("Fitting Logistic Regression model...")
    model_wrapper.fit(features=x_train, target=y_train)
    
    # 6. Save Locally
    joblib.dump(model_wrapper._model, MODEL_OUTPUT)
    print(f"Model serialized locally to {MODEL_OUTPUT}")

    # 7. Upload to Google Cloud Storage
    try:
        upload_to_gcs(BUCKET_NAME, MODEL_OUTPUT, MODEL_OUTPUT)
    except Exception as e:
        print(f"GCS Upload skipped/failed: {e}")

def upload_to_gcs(bucket_name: str, source_file: str, destination_blob: str) -> None:
    """Uploads a file to a Google Cloud Storage bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    
    print(f"Uploading to gs://{bucket_name}/{destination_blob}...")
    blob.upload_from_filename(source_file)
    print("Upload complete.")

if __name__ == "__main__":
    run_training_pipeline()