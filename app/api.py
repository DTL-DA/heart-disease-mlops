"""API REST para predecir enfermedad cardíaca con el modelo entrenado."""
import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

RUTA = Path(__file__).resolve().parent
model = joblib.load(RUTA / "model.joblib")
INFO = json.loads((RUTA / "model_info.json").read_text(encoding="utf-8"))

app = FastAPI(
    title="Heart Disease API",
    description="Predicción de enfermedad cardíaca (dataset Heart Failure Prediction).",
    version="1.0.0",
)


class Paciente(BaseModel):
    """Variables clínicas de un paciente, con los mismos nombres de heart.csv."""

    Age: int = Field(ge=18, le=110, description="Edad en años")
    Sex: Literal["M", "F"]
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"]
    RestingBP: int = Field(gt=0, le=300, description="Presión arterial en reposo, mm Hg")
    Cholesterol: int = Field(ge=0, le=1000, description="Colesterol, mg/dl. Use 0 si no fue medido")
    FastingBS: Literal[0, 1] = Field(description="1 si la glucemia en ayunas es mayor a 120 mg/dl")
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: int = Field(ge=40, le=250, description="Frecuencia cardíaca máxima")
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float = Field(ge=-5, le=10, description="Depresión del segmento ST")
    ST_Slope: Literal["Up", "Flat", "Down"]

    model_config = {"json_schema_extra": {"examples": [{
        "Age": 58, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140, "Cholesterol": 260,
        "FastingBS": 1, "RestingECG": "Normal", "MaxHR": 115, "ExerciseAngina": "Y",
        "Oldpeak": 2.0, "ST_Slope": "Flat"}]}}


@app.get("/")
def raiz():
    return {"servicio": "Heart Disease API", "documentacion": "/docs", "modelo": INFO["modelo"]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return INFO


@app.post("/predict")
def predict(data: Paciente):
    X = pd.DataFrame([data.model_dump()])
    proba = float(model.predict_proba(X)[0][1])
    return {"heart_disease_probability": round(proba, 4), "prediction": int(proba > 0.5)}
