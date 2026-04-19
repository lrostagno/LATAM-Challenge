# Model Logic
TARGET_COLUMN = "delay"
DELAY_THRESHOLD_MINUTES = 15
FEATURES_COLS = [
    "OPERA_Latin American Wings", "MES_7", "MES_10",
    "OPERA_Grupo LATAM", "MES_12", "TIPOVUELO_I",
    "MES_4", "MES_11", "OPERA_Sky Airline", "OPERA_Copa Air"
]
CATEGORICAL_COLS = ["OPERA", "TIPOVUELO", "MES"]

# Validation
VALID_OPERA = [
    'American Airlines', 'Air Canada', 'Air France', 'Aeromexico',
    'Aerolineas Argentinas', 'Austral', 'Avianca', 'Alitalia',
    'British Airways', 'Copa Air', 'Delta Air', 'Gol Trans', 'Iberia',
    'K.L.M.', 'Qantas Airways', 'United Airlines', 'Grupo LATAM',
    'Sky Airline', 'Latin American Wings', 'Plus Ultra Lineas Aereas',
    'JetSmart SPA', 'Oceanair Linhas Aereas', 'Lacsa'
]
VALID_TIPOVUELO = ["N", "I"]

# Infrastructure
import os
BUCKET_NAME = os.getenv("MODEL_BUCKET", "bucket-latam-challenge")
MODEL_FILE_NAME = "model.joblib"
LOCAL_MODEL_PATH = "/tmp/model.joblib"