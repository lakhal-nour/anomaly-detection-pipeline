# src/api/main.py
from fastapi import FastAPI, HTTPException, Security, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.schemas       import NetworkLog, PredictionResponse, HealthResponse
from api.model_loader  import model_loader
from api.security      import verify_api_key, check_rate_limit

app = FastAPI(
    title="SIEM Anomaly Detection API",
    description="Pipeline DevSecOps/MLOps — Détection d'anomalies réseau",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.on_event("startup")
async def startup_event():
    os.chdir("/app")
    model_loader.load()
    print("API v2.0 démarrée avec contrôle d'accès ✅")

@app.get("/health")
async def health():
    """Endpoint public — pas d'authentification requise."""
    return {
        "status":   "healthy" if model_loader.is_loaded else "degraded",
        "model":    model_loader.model_type,
        "security": "API Key + Rate Limiting actifs"
    }

@app.post("/detect", response_model=PredictionResponse)
async def detect_anomaly(
    log:     NetworkLog,
    request: Request,
    api_key: str = Security(verify_api_key)
):
    """
    Endpoint protégé — requiert une clé API valide.
    Header requis : X-API-Key: siem-secret-key-2024
    """
    # Rate limiting
    check_rate_limit(request)

    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    try:
        features = [
            log.flow_duration, log.tot_fwd_pkts, log.tot_bwd_pkts,
            log.totlen_fwd_pkts, log.totlen_bwd_pkts,
            log.fwd_pkt_len_max, log.fwd_pkt_len_min, log.fwd_pkt_len_mean,
            log.bwd_pkt_len_max, log.bwd_pkt_len_min,
            log.flow_byts_s, log.flow_pkts_s, log.psh_flag_cnt,
            log.ack_flag_cnt, log.pkt_len_min, log.pkt_len_max,
            log.pkt_len_mean, log.pkt_len_std, log.pkt_len_var,
            log.init_win_bytes_forward, log.init_win_bytes_backward,
            log.min_seg_size_forward
        ]
        while len(features) < 52:
            features.append(0.0)

        result = model_loader.predict(features)
        msg    = f"ANOMALIE DÉTECTÉE — score={result['anomaly_score']:.3f}" \
                 if result["is_anomaly"] \
                 else f"Trafic normal — score={result['anomaly_score']:.3f}"

        return PredictionResponse(
            is_anomaly    = result["is_anomaly"],
            anomaly_score = result["anomaly_score"],
            confidence    = result["confidence"],
            model_used    = result["model_used"],
            message       = msg
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Métriques Prometheus ─────────────────────────────────────
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse
import time

# Compteurs
REQUESTS_TOTAL    = Counter('siem_requests_total', 'Total requêtes', ['endpoint'])
ANOMALIES_TOTAL   = Counter('siem_anomalies_total', 'Total anomalies détectées')
LATENCY_HISTOGRAM = Histogram('siem_request_duration_seconds', 'Latence des requêtes')

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Endpoint Prometheus — expose les métriques."""
    return generate_latest()
