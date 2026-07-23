"""FastAPI entrypoint for the predictor service.

Deployed as a Vercel "service" (see repo-root vercel.json) bound internally to the
Next.js frontend service -- never exposed on the public internet directly. The
frontend's /api/predict route calls POST /predict here over that internal binding,
forwarding the caller's own Anthropic API key in the X-Api-Key header.

For local development without Vercel, run this with:
    uvicorn app:app --reload --port 8000
and point the frontend at it with PREDICTOR_URL=http://localhost:8000.
"""
from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from predictor.agent import predict_match

app = FastAPI()


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    target_year: int


@app.post("/predict")
def predict(req: PredictRequest, x_api_key: str | None = Header(default=None)) -> dict:
    try:
        return predict_match(req.home_team, req.away_team, req.target_year, api_key=x_api_key)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the caller as a clean 500
        raise HTTPException(status_code=500, detail=str(exc)) from exc
