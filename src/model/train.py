# src/model/train.py
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from preprocess import prepare_data

def train_model(X_normal: np.ndarray, contamination: float = 0.01):
    """Entraîne Isolation Forest sur le trafic normal uniquement."""
    print(f"\nEntraînement sur {len(X_normal):,} connexions normales...")
    
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_normal)
    print("Modèle entraîné ✅")
    return model

def evaluate_model(model, X_normal, X_attack):
    """Évalue le modèle sur normal + attaques mélangés."""
    print("\nÉvaluation du modèle...")

    # Prédictions : -1 = anomalie, 1 = normal
    pred_normal = model.predict(X_normal)
    pred_attack = model.predict(X_attack)

    # Construire les vrais labels et prédictions
    y_true = np.array([0] * len(X_normal) + [1] * len(X_attack))
    y_pred = np.array(
        [0 if p == 1 else 1 for p in pred_normal] +
        [0 if p == 1 else 1 for p in pred_attack]
    )

    # Scores d'anomalie
    scores_normal = -model.score_samples(X_normal)
    scores_attack = -model.score_samples(X_attack)
    y_scores = np.concatenate([scores_normal, scores_attack])

    # Métriques
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)

    print(f"\n{'='*40}")
    print(f"F1-Score  : {f1:.3f}")
    print(f"AUC-ROC   : {auc:.3f}")
    print(f"{'='*40}")
    print("\nRapport détaillé :")
    print(classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Attaque']
    ))
    print("Matrice de confusion :")
    print(confusion_matrix(y_true, y_pred))

    return f1, auc

def save_model(model, path: str = "models/isolation_forest.pkl"):
    """Sauvegarde le modèle entraîné."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\nModèle sauvegardé : {path} ✅")

if __name__ == "__main__":
    # 1. Préparer les données
    X_normal, X_attack, labels, features = prepare_data(
        "data/raw/cicids2017_cleaned.csv",
        nrows=50000
    )

    # 2. Entraîner
    model = train_model(X_normal, contamination=0.01)

    # 3. Évaluer
    f1, auc = evaluate_model(model, X_normal, X_attack)

    # 4. Sauvegarder
    save_model(model)

    print("\nPipeline ML terminé ✅")
    print(f"F1={f1:.3f} | AUC={auc:.3f}")
