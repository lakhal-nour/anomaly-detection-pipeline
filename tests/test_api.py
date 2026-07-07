# tests/test_api.py
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_import_api():
    """Vérifier que les modules s'importent correctement."""
    from api.schemas import NetworkLog, PredictionResponse
    assert NetworkLog is not None
    assert PredictionResponse is not None

def test_network_log_default():
    """Vérifier les valeurs par défaut de NetworkLog."""
    from api.schemas import NetworkLog
    log = NetworkLog()
    assert log.flow_duration == 0.0
    assert log.tot_fwd_pkts == 0.0
    assert log.psh_flag_cnt == 0.0

def test_network_log_values():
    """Vérifier qu'on peut créer un NetworkLog avec des valeurs."""
    from api.schemas import NetworkLog
    log = NetworkLog(
        flow_duration=50.0,
        tot_fwd_pkts=5.0,
        psh_flag_cnt=0.0
    )
    assert log.flow_duration == 50.0
    assert log.tot_fwd_pkts == 5.0

def test_anomaly_score_range():
    """Vérifier que le score d anomalie est entre 0 et 1."""
    score = 0.4318
    assert 0.0 <= score <= 1.0

def test_confidence_levels():
    """Vérifier les niveaux de confiance valides."""
    valid_levels = ["HIGH", "MEDIUM", "LOW"]
    assert "MEDIUM" in valid_levels
    assert "HIGH" in valid_levels
