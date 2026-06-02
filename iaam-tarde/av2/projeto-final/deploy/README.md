# Deploy — Classificador de Ocorrencias NEAC-AL

API FastAPI que serve o pipeline treinado (ARMA / CVLI / DROGA).

## Build e execucao

    docker build -t neac-api:1.0 .
    docker run -p 8000:8000 neac-api:1.0

Docs OpenAPI: http://localhost:8000/docs

## Exemplo

    curl -X POST http://localhost:8000/prever -H "Content-Type: application/json" -d '{
      "MES_FATO": 6, "DIA_FATO": 15, "HORA_FATO": 20,
      "LONGITUDE": -35.73, "LATITUDE": -9.66,
      "DIA_SEMANA_FATO": "Sex", "TURNO": "Noite",
      "CIDADE_FATO": "Maceio", "RISP": "1a RISP", "AISP": "1a AISP",
      "AMBIENTE": "Via Publica"
    }'
