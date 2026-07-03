# src/model/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(filepath: str, nrows: int = 50000) -> pd.DataFrame:
    print(f"Chargement de {nrows:,} lignes...")
    df = pd.read_csv(filepath, nrows=nrows)
    print(f"Dataset chargé : {df.shape}")
    return df

def get_feature_columns(df: pd.DataFrame):
    label_col = [c for c in df.columns if 'attack type' in c.lower()][0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Colonne label détectée : '{label_col}'")
    print(f"Nombre de features numériques : {len(numeric_cols)}")
    return numeric_cols, label_col

def clean_data(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    X = df[feature_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    print(f"Après nettoyage : {X.shape[0]:,} lignes")
    return X

def normalize_data(X, scaler_path: str = None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"Scaler sauvegardé : {scaler_path}")
    return X_scaled, scaler

def prepare_data(filepath: str, nrows: int = 50000):
    # 1. Charger
    df = load_data(filepath, nrows)

    # 2. Identifier les colonnes
    feature_cols, label_col = get_feature_columns(df)

    # 3. Labels
    labels = df[label_col].copy()

    # 4. Nettoyer
    X = clean_data(df, feature_cols)
    labels = labels.loc[X.index]

    # DEBUG — voir les valeurs réelles
    print(f"\nValeurs uniques dans '{label_col}' :")
    print(labels.value_counts())

    # 5. Séparer normal et attaques
    normal_mask = labels == 'Normal Traffic'
    X_normal = X[normal_mask]
    X_attack = X[~normal_mask]

    print(f"\nTrafic normal  : {len(X_normal):,} lignes")
    print(f"Trafic attaque : {len(X_attack):,} lignes")

    if len(X_normal) == 0:
        raise ValueError("Aucune ligne normale trouvée — vérifier le nom exact dans la colonne")

    # 6. Normaliser
    X_normal_scaled, scaler = normalize_data(
        X_normal,
        scaler_path="models/scaler.pkl"
    )
    X_attack_scaled = scaler.transform(X_attack)

    return X_normal_scaled, X_attack_scaled, labels, feature_cols

if __name__ == "__main__":
    X_normal, X_attack, labels, features = prepare_data(
        "data/raw/cicids2017_cleaned.csv",
        nrows=50000
    )
    print("\nPréprocessing terminé ✅")
    print(f"Features utilisées : {len(features)}")
