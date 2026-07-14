# src/api/security.py
# Contrôle d'accès : API Key + Rate Limiting

from fastapi import HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from collections import defaultdict
from datetime import datetime, timedelta
import os, time

# API Key simple
API_KEY        = os.getenv("API_KEY", "siem-secret-key-2024")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate limiting : max 100 requêtes par minute par IP
RATE_LIMIT     = 100
WINDOW_SECONDS = 60
request_counts = defaultdict(list)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Vérifie que la requête contient une clé API valide."""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Clé API invalide ou manquante — accès refusé"
        )
    return api_key

def check_rate_limit(request: Request):
    """Limite le nombre de requêtes par IP."""
    client_ip = request.client.host
    now       = time.time()
    window    = now - WINDOW_SECONDS

    # Nettoyer les anciennes requêtes
    request_counts[client_ip] = [
        t for t in request_counts[client_ip] if t > window
    ]

    if len(request_counts[client_ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Trop de requêtes — limite : {RATE_LIMIT}/minute"
        )

    request_counts[client_ip].append(now)
