# src/api/schemas.py
from pydantic import BaseModel
from typing import Optional

class NetworkLog(BaseModel):
    """
    Structure d'une connexion réseau envoyée à l'API.
    Correspond aux features du dataset CICIDS2017.
    """
    # Features principales
    flow_duration:          float = 0.0
    tot_fwd_pkts:           float = 0.0
    tot_bwd_pkts:           float = 0.0
    totlen_fwd_pkts:        float = 0.0
    totlen_bwd_pkts:        float = 0.0
    fwd_pkt_len_max:        float = 0.0
    fwd_pkt_len_min:        float = 0.0
    fwd_pkt_len_mean:       float = 0.0
    bwd_pkt_len_max:        float = 0.0
    bwd_pkt_len_min:        float = 0.0
    flow_byts_s:            float = 0.0
    flow_pkts_s:            float = 0.0
    psh_flag_cnt:           float = 0.0
    ack_flag_cnt:           float = 0.0
    pkt_len_min:            float = 0.0
    pkt_len_max:            float = 0.0
    pkt_len_mean:           float = 0.0
    pkt_len_std:            float = 0.0
    pkt_len_var:            float = 0.0
    init_win_bytes_forward: float = 0.0
    init_win_bytes_backward:float = 0.0
    min_seg_size_forward:   float = 0.0

class PredictionResponse(BaseModel):
    """Réponse retournée par l'API."""
    is_anomaly:      bool
    anomaly_score:   float
    confidence:      str        # "HIGH", "MEDIUM", "LOW"
    model_used:      str
    message:         str

class HealthResponse(BaseModel):
    """Réponse du endpoint de santé."""
    status:          str
    model_loaded:    bool
    model_name:      str
    version:         str
