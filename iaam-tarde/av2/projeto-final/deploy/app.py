"""API de inferencia — classificacao da natureza de ocorrencia criminal (ARMA/CVLI/DROGA)."""
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="NEAC-AL — Classificador de Ocorrencias",
    description="Preve a natureza da ocorrencia (ARMA, CVLI, DROGA) a partir de atributos espaco-temporais.",
    version="1.0.0",
)

modelo = joblib.load("modelo.joblib")
with open("metadata.json", encoding="utf-8") as f:
    META = json.load(f)


class Ocorrencia(BaseModel):
    MES_FATO: int = Field(..., ge=1, le=12)
    DIA_FATO: int = Field(..., ge=1, le=31)
    HORA_FATO: int = Field(..., ge=0, le=23)
    LONGITUDE: float
    LATITUDE: float
    DIA_SEMANA_FATO: str
    TURNO: str
    CIDADE_FATO: str
    RISP: str
    AISP: str
    AMBIENTE: str


@app.get("/health")
def health():
    return {"status": "ok", "classes": META["classes"]}


@app.post("/prever")
def prever(o: Ocorrencia):
    linha = pd.DataFrame([o.dict()])
    classe = modelo.predict(linha)[0]
    probs = modelo.predict_proba(linha)[0]
    return {
        "classe_prevista": str(classe),
        "confianca": float(max(probs)),
        "probabilidades": {c: float(p) for c, p in zip(modelo.classes_, probs)},
    }
