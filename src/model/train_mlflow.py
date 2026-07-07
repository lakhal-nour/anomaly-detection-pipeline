# src/model/train_mlflow.py
import numpy as np
import joblib, os, sys
import mlflow
import mlflow.sklearn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import prepare_data
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# Configurer MLflow
os.chdir("/home/nour/anomaly-detection-pipeline")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("anomaly-detection-cicids2017")

def run_isolation_forest(X_normal, X_attack):
    """Loggue Isolation Forest dans MLflow."""
    with mlflow.start_run(run_name="IsolationForest"):

        # Paramètres
        mlflow.log_param("model_type",    "IsolationForest")
        mlflow.log_param("n_estimators",  200)
        mlflow.log_param("contamination", 0.30)
        mlflow.log_param("n_normal",      len(X_normal))
        mlflow.log_param("n_attack",      len(X_attack))
        mlflow.log_param("dataset",       "CICIDS2017_300k")

        # Entraînement
        model = IsolationForest(
            n_estimators=200,
            contamination=0.30,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_normal)

        # Évaluation
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
        cm  = confusion_matrix(y_true, best_pred)

        # Logger les métriques
        mlflow.log_metric("f1_score",        best_f1)
        mlflow.log_metric("auc_roc",         auc)
        mlflow.log_metric("threshold",       best_t)
        mlflow.log_metric("true_positives",  int(cm[1][1]))
        mlflow.log_metric("false_negatives", int(cm[1][0]))
        mlflow.log_metric("false_positives", int(cm[0][1]))

        # Sauvegarder le modèle dans MLflow
        mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id
        print(f"[IF] F1={best_f1:.3f} | AUC={auc:.3f} | run_id={run_id[:8]}")

    return model, best_f1, auc, best_t

def run_random_forest(X_normal, X_attack):
    """Loggue Random Forest dans MLflow."""
    with mlflow.start_run(run_name="RandomForest"):

        mlflow.log_param("model_type",     "RandomForest")
        mlflow.log_param("n_estimators",   100)
        mlflow.log_param("max_depth",      15)
        mlflow.log_param("min_samples_leaf", 10)
        mlflow.log_param("class_weight",   "balanced")
        mlflow.log_param("dataset",        "CICIDS2017_300k")

        X_all = np.vstack([X_normal, X_attack])
        y_all = np.array([0]*len(X_normal) + [1]*len(X_attack))

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size=0.20,
            random_state=42,
            stratify=y_all
        )

        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size",  len(X_test))

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
        cm       = confusion_matrix(y_test, y_pred)

        # Cross-validation
        cv = cross_val_score(
            RandomForestClassifier(
                n_estimators=50, max_depth=15,
                min_samples_leaf=10,
                class_weight='balanced', random_state=42
            ),
            X_all, y_all, cv=5, scoring='f1', n_jobs=-1
        )

        # Logger tout
        mlflow.log_metric("f1_score",        f1)
        mlflow.log_metric("auc_roc",         auc)
        mlflow.log_metric("cv_f1_mean",      cv.mean())
        mlflow.log_metric("cv_f1_std",       cv.std())
        mlflow.log_metric("true_positives",  int(cm[1][1]))
        mlflow.log_metric("false_negatives", int(cm[1][0]))
        mlflow.log_metric("false_positives", int(cm[0][1]))

        mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id
        print(f"[RF] F1={f1:.3f} | AUC={auc:.3f} | CV={cv.mean():.3f} | run_id={run_id[:8]}")

    return model, f1, auc

if __name__ == "__main__":
    print("Chargement des données...")
    X_normal, X_attack, labels, features = prepare_data(
        "data/raw/cicids2017_cleaned.csv",
        nrows=300000
    )

    print(f"Répartition : {len(X_normal):,} normaux | {len(X_attack):,} attaques")
    print("\nDémarrage des expériences MLflow...")

    if_model, f1_if, auc_if, threshold = run_isolation_forest(X_normal, X_attack)
    rf_model, f1_rf, auc_rf            = run_random_forest(X_normal, X_attack)

    print("\n" + "="*50)
    print("EXPÉRIENCES LOGGÉES DANS MLFLOW ✅")
    print("="*50)
    print(f"Isolation Forest : F1={f1_if:.3f} | AUC={auc_if:.3f}")
    print(f"Random Forest    : F1={f1_rf:.3f} | AUC={auc_rf:.3f}")
    print("\nPour visualiser :")
    print("mlflow ui --host 0.0.0.0 --port 5000")
    print("Puis ouvre : http://localhost:5000")
    print("="*50)
