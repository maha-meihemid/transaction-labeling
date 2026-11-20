import hashlib
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.transaction_labeling.models import predict_minor
from src.transaction_labeling.schemas import Transaction


ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = ROOT / "models" / "transaction_labeler.joblib"
MANIFEST_FILE = ROOT / "models" / "model_manifest.json"
LOGGER = logging.getLogger(__name__)

if not MODEL_FILE.is_file() or not MANIFEST_FILE.is_file():
    raise RuntimeError("Modèle introuvable. Lancez `python -m src.transaction_labeling.train`.")

manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
model_hash = hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest()
if model_hash != manifest.get("model_sha256"):
    raise RuntimeError("L’artefact du modèle ne correspond pas à son manifeste.")
bundle = joblib.load(MODEL_FILE)


app = FastAPI(title="Transaction Labeling API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "model": manifest["model_type"]}


@app.post("/predict")
def predict(transaction: Transaction):
    row = pd.DataFrame([transaction.model_dump()])
    try:
        minor_category = predict_minor(bundle, row).iloc[0]
        major_mapping = bundle["major_category_mapping"]
        if minor_category not in major_mapping:
            raise ValueError(f"Catégorie fine sans correspondance : {minor_category}.")
        major_category = major_mapping[minor_category]
    except Exception as error:
        LOGGER.exception("Échec de prédiction pour le modèle %s", manifest["model_type"])
        raise HTTPException(
            status_code=500,
            detail="Une erreur interne est survenue pendant la prédiction.",
        ) from error

    return {
        "minor_category": minor_category,
        "major_category": major_category,
    }
