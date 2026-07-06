# src/model/train.py
import numpy as np
import joblib, os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import prepare_data
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

def train_isolation_forest(X_normal, X_attack):
    print("\n[1] Isolation Forest (non supervisé)...")
    model = IsolationForest(
        n_estimators=200,
        contamination=0.30,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_normal)

    scores_normal = -model.score_samples(X_normal)
    scores_attack = -model.score_samples(X_attack)
    y_true        = np.array([0]*len(X_normal) + [1]*len(X_attack))
    y_scores      = np.concatenate([scores_normal, scores_attack])

    best_f1, best_pred, best_t = 0, None, 0
    for pct in range(50, 99, 5):
        t      = np.percentile(y_scores, pct)
        y_pred = (y_scores > t).astype(int)
        f1     = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_pred, best_t = f1, y_pred, t

    auc = roc_auc_score(y_true, y_scores)
    print(f"  F1={best_f1:.3f} | AUC={auc:.3f}")
    print(classification_report(
        y_true, best_pred,
        target_names=['Normal','Attaque'], zero_division=0
    ))
    return model, best_f1, auc, best_t

def train_random_forest(X_normal, X_attack):
    print("\n[2] Random Forest (supervisé — split stratifié)...")

    X_all = np.vstack([X_normal, X_attack])
    y_all = np.array([0]*len(X_normal) + [1]*len(X_attack))

    # Split stratifié — garantit les deux classes dans train ET test
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=0.20,
        random_state=42,
        stratify=y_all    # ← garantit proportion égale dans train/test
    )

    print(f"  Train : {len(X_train):,} | Test : {len(X_test):,}")
    print(f"  Attaques train : {y_train.sum():,} | Attaques test : {y_test.sum():,}")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    f1       = f1_score(y_test, y_pred, zero_division=0)
    auc      = roc_auc_score(y_test, y_scores)

    print(f"\n  F1={f1:.3f} | AUC={auc:.3f}")
    print(classification_report(
        y_test, y_pred,
        target_names=['Normal','Attaque'], zero_division=0
    ))
    print("  Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    # Validation croisée
    print("\n  Validation croisée (5 folds)...")
    cv = cross_val_score(
        RandomForestClassifier(
            n_estimators=50, max_depth=15,
            min_samples_leaf=10,
            class_weight='balanced', random_state=42
        ),
        X_all, y_all, cv=5, scoring='f1', n_jobs=-1
    )
    print(f"  F1 par fold : {[f'{s:.3f}' for s in cv]}")
    print(f"  F1 moyen    : {cv.mean():.3f} (+/- {cv.std():.3f})")

    return model, f1, auc

def save_models(if_model, rf_model, f1_if, f1_rf, threshold):
    os.makedirs("models", exist_ok=True)
    joblib.dump({'model': if_model, 'type': 'IsolationForest',
                 'threshold': threshold}, "models/isolation_forest.pkl")
    joblib.dump({'model': rf_model, 'type': 'RandomForest'},
                "models/random_forest.pkl")
    best = "RandomForest" if f1_rf > f1_if else "IsolationForest"
    print(f"\nModèles sauvegardés ✅ — Meilleur : {best}")
    return best

if __name__ == "__main__":
    X_normal, X_attack, labels, features = prepare_data(
        "data/raw/cicids2017_cleaned.csv",
        nrows=300000
    )
    print(f"\nRépartition : {len(X_normal):,} normaux | {len(X_attack):,} attaques")

    if_model, f1_if, auc_if, threshold = train_isolation_forest(
        X_normal, X_attack
    )
    rf_model, f1_rf, auc_rf = train_random_forest(
        X_normal, X_attack
    )
    save_models(if_model, rf_model, f1_if, f1_rf, threshold)

    print("\n" + "="*50)
    print("RÉSULTATS FINAUX")
    print("="*50)
    print(f"Isolation Forest : F1={f1_if:.3f} | AUC={auc_if:.3f}")
    print(f"Random Forest    : F1={f1_rf:.3f} | AUC={auc_rf:.3f}")
    print("="*50)
