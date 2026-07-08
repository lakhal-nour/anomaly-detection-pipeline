# src/api/model_integrity.py
# Contre-mesure contre la désérialisation malveillante (CVE-2024-37058, CVE-2024-37065)

import hashlib
import json
import os

HASHES_FILE = "models/hashes.json"

def compute_hash(filepath: str) -> str:
    """Calcule le SHA-256 d'un fichier."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def generate_hashes():
    """Génère et sauvegarde les hashs de référence des modèles."""
    models = {
        "isolation_forest": "models/isolation_forest.pkl",
        "random_forest":    "models/random_forest.pkl",
        "scaler":           "models/scaler.pkl"
    }
    hashes = {}
    for name, path in models.items():
        if os.path.exists(path):
            hashes[name] = {
                "path":   path,
                "sha256": compute_hash(path)
            }
            print(f"Hash {name}: {hashes[name]['sha256'][:16]}...")

    with open(HASHES_FILE, "w") as f:
        json.dump(hashes, f, indent=2)

    print(f"\nHashs sauvegardés dans {HASHES_FILE} ✅")
    return hashes

def verify_integrity() -> bool:
    """
    Vérifie l'intégrité des modèles avant chargement.
    Protection contre CVE-2024-37058 et CVE-2024-37065.
    Retourne True si tout est intact, False si compromis.
    """
    if not os.path.exists(HASHES_FILE):
        print("AVERTISSEMENT: Fichier de hashs manquant — génération...")
        generate_hashes()
        return True

    with open(HASHES_FILE) as f:
        reference_hashes = json.load(f)

    all_ok = True
    for name, info in reference_hashes.items():
        path = info["path"]
        if not os.path.exists(path):
            print(f"ERREUR: Modèle manquant : {path}")
            all_ok = False
            continue

        current_hash = compute_hash(path)
        if current_hash != info["sha256"]:
            print(f"ALERTE SÉCURITÉ: Modèle {name} modifié !")
            print(f"  Hash attendu  : {info['sha256'][:32]}...")
            print(f"  Hash actuel   : {current_hash[:32]}...")
            all_ok = False
        else:
            print(f"Intégrité OK : {name} ✅")

    return all_ok

if __name__ == "__main__":
    print("Génération des hashs de référence...")
    generate_hashes()
    print("\nVérification de l'intégrité...")
    ok = verify_integrity()
    print(f"\nRésultat : {'✅ Tous les modèles sont intègres' if ok else '❌ COMPROMISSION DÉTECTÉE'}")
