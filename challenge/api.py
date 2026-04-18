import os
import logging
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from google.cloud import storage
from challenge.model import DelayModel

logger = logging.getLogger("uvicorn.error")

app = FastAPI()


@app.on_event("startup")
async def startup():
    """
    Runs when the API starts.
    Downloads the model from GCS and loads it into memory.
    """
    logger.info("STARTUP: Initializing model")

    app.state.model = DelayModel()

    bucket_name = os.getenv("MODEL_BUCKET", "bucket-latam-challenge")
    model_file = "model.joblib"
    local_path = "/tmp/model.joblib"

    try:
        logger.info(f"Connecting to GCS bucket: {bucket_name}")

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(model_file)

        blob.download_to_filename(local_path)
        logger.info(f"Model downloaded to {local_path}")

        app.state.model.load(local_path)
        logger.info("Model loaded into memory successfully")

    except Exception as e:
        logger.error(f"CRITICAL: Startup failed to load model: {str(e)}")
        raise e


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(data: dict, request: Request) -> dict:
    if not hasattr(request.app.state, "model"):
        return {"error": "Model not loaded"}

    model = request.app.state.model

    flights = data.get("flights", [])
    if not flights:
        raise HTTPException(status_code=400, detail="Missing flights")

    df = pd.DataFrame(flights)

    try:
        if "MES" in df and not df["MES"].between(1, 12).all():
            raise ValueError("Invalid MES")

        if "TIPOVUELO" in df and not df["TIPOVUELO"].isin(["N", "I"]).all():
            raise ValueError("Invalid TIPOVUELO")

        if "OPERA" in df and not df["OPERA"].isin([
            "Aerolineas Argentinas",
            "LATAM Airlines Group",
            "Sky Airline",
            "JetSmart SPA"
        ]).all():
            raise ValueError("Invalid OPERA")

        features = model.preprocess(df)
        preds = model.predict(features)

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid input")

    return {"predict": preds}