# src/api/main.py
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import os, sys, time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.schemas      import NetworkLog, PredictionResponse, HealthResponse
from api.model_loader import model_loader
from api.security     import verify_api_key, check_rate_limit

# ── Métriques Prometheus ──────────────────────────────────────
REQUESTS_TOTAL  = Counter(
    'siem_requests_total',
    'Total requêtes reçues',
    ['endpoint', 'status']
)
ANOMALIES_TOTAL = Counter(
    'siem_anomalies_total',
    'Total anomalies détectées'
)
NORMAL_TOTAL    = Counter(
    'siem_normal_total',
    'Total trafic normal détecté'
)
LATENCY         = Histogram(
    'siem_request_duration_seconds',
    'Latence des requêtes en secondes'
)
MODEL_SCORE     = Gauge(
    'siem_last_anomaly_score',
    'Dernier score d anomalie calculé'
)

app = FastAPI(
    title="SIEM Anomaly Detection API",
    description="Pipeline DevSecOps/MLOps — Détection d anomalies réseau",
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
    print("API v2.0 démarrée ✅")

@app.get("/health")
async def health():
    REQUESTS_TOTAL.labels(endpoint="health", status="200").inc()
    return {
        "status":   "healthy" if model_loader.is_loaded else "degraded",
        "model":    model_loader.model_type,
        "security": "API Key + Rate Limiting actifs"
    }

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Endpoint Prometheus."""
    return generate_latest()

@app.post("/detect", response_model=PredictionResponse)
async def detect_anomaly(
    log:     NetworkLog,
    request: Request,
    api_key: str = Security(verify_api_key)
):
    start_time = time.time()

    # Rate limiting
    check_rate_limit(request)

    if not model_loader.is_loaded:
        REQUESTS_TOTAL.labels(endpoint="detect", status="503").inc()
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

        # ── Incrémenter les métriques ──────────────────────────
        duration = time.time() - start_time
        LATENCY.observe(duration)
        REQUESTS_TOTAL.labels(endpoint="detect", status="200").inc()
        MODEL_SCORE.set(result["anomaly_score"])

        if result["is_anomaly"]:
            ANOMALIES_TOTAL.inc()
            msg = f"ANOMALIE DÉTECTÉE — score={result['anomaly_score']:.3f}"
        else:
            NORMAL_TOTAL.inc()
            msg = f"Trafic normal — score={result['anomaly_score']:.3f}"

        return PredictionResponse(
            is_anomaly    = result["is_anomaly"],
            anomaly_score = result["anomaly_score"],
            confidence    = result["confidence"],
            model_used    = result["model_used"],
            message       = msg
        )

    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint="detect", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))
