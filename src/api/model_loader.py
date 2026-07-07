# src/api/model_loader.py
import joblib
import numpy as np
import os

class ModelLoader:
    """
    Charge et gère le modèle ML.
    Singleton — chargé une seule fois au démarrage de l'API.
    """
    def __init__(self):
        self.model      = None
        self.scaler     = None
        self.threshold  = None
        self.model_type = None
        self.is_loaded  = False

    def load(self):
        """Charge le modèle et le scaler depuis le dossier models/."""
        try:
            # Charger Isolation Forest (modèle non supervisé)
            model_path  = "models/isolation_forest.pkl"
            scaler_path = "models/scaler.pkl"

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modèle introuvable : {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler introuvable : {scaler_path}")

            data            = joblib.load(model_path)
            self.model      = data['model']
            self.threshold  = data.get('threshold', 0.45)
            self.model_type = data.get('type', 'IsolationForest')
            self.scaler     = joblib.load(scaler_path)
            self.is_loaded  = True

            print(f"Modèle chargé : {self.model_type}")
            print(f"Seuil anomalie : {self.threshold:.4f}")

        except Exception as e:
            print(f"Erreur chargement modèle : {e}")
            self.is_loaded = False

    def predict(self, features: list) -> dict:
        """
        Prédit si une connexion est une anomalie.
        Retourne le score et la décision.
        """
        if not self.is_loaded:
            raise RuntimeError("Modèle non chargé")

        # Normaliser les features
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Score d'anomalie (plus c'est élevé, plus c'est suspect)
        score = float(-self.model.score_samples(X_scaled)[0])

        # Décision basée sur le seuil
        is_anomaly = score > self.threshold

        # Niveau de confiance
        diff = abs(score - self.threshold)
        if diff > 0.15:
            confidence = "HIGH"
        elif diff > 0.05:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return {
            "is_anomaly":    is_anomaly,
            "anomaly_score": round(score, 4),
            "confidence":    confidence,
            "model_used":    self.model_type
        }

# Instance globale — chargée une seule fois
model_loader = ModelLoader()
