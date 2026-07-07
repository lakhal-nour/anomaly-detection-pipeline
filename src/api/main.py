# src/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.schemas import NetworkLog, PredictionResponse, HealthResponse
from api.model_loader import model_loader

# Créer l'application FastAPI
app = FastAPI(
    title="SIEM Anomaly Detection API",
    description="API de détection d anomalies réseau bancaires — Pipeline DevSecOps/MLOps",
    version="1.0.0"
)

# CORS — permettre les requêtes depuis n'importe quelle origine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.on_event("startup")
async def startup_event():
    """Charger le modèle au démarrage de l'API."""
    print("Démarrage de l'API...")
    os.chdir("/home/nour/anomaly-detection-pipeline")
    model_loader.load()
    print("API prête ✅")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Vérifier que l'API fonctionne."""
    return HealthResponse(
        status="ok" if model_loader.is_loaded else "degraded",
        model_loaded=model_loader.is_loaded,
        model_name="SIEM_Detector_Model",
        version="1.0.0"
    )

@app.get("/health")
async def health():
    """Endpoint de santé pour Kubernetes/Docker."""
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "healthy", "model": model_loader.model_type}

@app.post("/detect", response_model=PredictionResponse)
async def detect_anomaly(log: NetworkLog):
    """
    Analyser une connexion réseau et détecter si c'est une anomalie.

    Envoie les features d'une connexion réseau et reçois :
    - is_anomaly : True si c'est une attaque détectée
    - anomaly_score : score entre 0 et 1
    - confidence : HIGH / MEDIUM / LOW
    """
    if not model_loader.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible"
        )

    try:
        # Extraire les features dans le bon ordre
        features = [
            log.flow_duration,
            log.tot_fwd_pkts,
            log.tot_bwd_pkts,
            log.totlen_fwd_pkts,
            log.totlen_bwd_pkts,
            log.fwd_pkt_len_max,
            log.fwd_pkt_len_min,
            log.fwd_pkt_len_mean,
            log.bwd_pkt_len_max,
            log.bwd_pkt_len_min,
            log.flow_byts_s,
            log.flow_pkts_s,
            log.psh_flag_cnt,
            log.ack_flag_cnt,
            log.pkt_len_min,
            log.pkt_len_max,
            log.pkt_len_mean,
            log.pkt_len_std,
            log.pkt_len_var,
            log.init_win_bytes_forward,
            log.init_win_bytes_backward,
            log.min_seg_size_forward
        ]

        # Compléter avec des zéros pour les features manquantes
        # Le modèle attend 52 features
        while len(features) < 52:
            features.append(0.0)

        # Prédiction
        result = model_loader.predict(features)

        # Message lisible
        if result["is_anomaly"]:
            msg = f"ANOMALIE DÉTECTÉE — score={result['anomaly_score']:.3f}"
        else:
            msg = f"Trafic normal — score={result['anomaly_score']:.3f}"

        return PredictionResponse(
            is_anomaly=result["is_anomaly"],
            anomaly_score=result["anomaly_score"],
            confidence=result["confidence"],
            model_used=result["model_used"],
            message=msg
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch")
async def detect_batch(logs: list[NetworkLog]):
    """Analyser plusieurs connexions en une seule requête."""
    if len(logs) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 1000 connexions par batch"
        )
    results = []
    for log in logs:
        result = await detect_anomaly(log)
        results.append(result)
    return {"total": len(results), "anomalies": sum(1 for r in results if r.is_anomaly), "results": results}
