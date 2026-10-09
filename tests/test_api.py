"""Pruebas automáticas de la API. Las ejecuta GitHub Actions en cada push."""
from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)

ALTO_RIESGO = {"Age": 62, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 150, "Cholesterol": 270,
               "FastingBS": 1, "RestingECG": "ST", "MaxHR": 105, "ExerciseAngina": "Y",
               "Oldpeak": 2.5, "ST_Slope": "Flat"}
BAJO_RIESGO = {"Age": 35, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 118, "Cholesterol": 190,
               "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 175, "ExerciseAngina": "N",
               "Oldpeak": 0.0, "ST_Slope": "Up"}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_devuelve_probabilidad_valida():
    r = client.post("/predict", json=ALTO_RIESGO)
    assert r.status_code == 200
    cuerpo = r.json()
    assert 0.0 <= cuerpo["heart_disease_probability"] <= 1.0
    assert cuerpo["prediction"] in (0, 1)


def test_el_modelo_ordena_bien_dos_perfiles_opuestos():
    alto = client.post("/predict", json=ALTO_RIESGO).json()
    bajo = client.post("/predict", json=BAJO_RIESGO).json()
    assert alto["heart_disease_probability"] > bajo["heart_disease_probability"]
    assert alto["prediction"] == 1
    assert bajo["prediction"] == 0


def test_colesterol_no_medido_no_rompe_la_api():
    r = client.post("/predict", json={**ALTO_RIESGO, "Cholesterol": 0})
    assert r.status_code == 200


def test_rechaza_categoria_desconocida():
    r = client.post("/predict", json={**ALTO_RIESGO, "Sex": "X"})
    assert r.status_code == 422


def test_rechaza_campo_faltante():
    incompleto = {k: v for k, v in ALTO_RIESGO.items() if k != "Age"}
    assert client.post("/predict", json=incompleto).status_code == 422
