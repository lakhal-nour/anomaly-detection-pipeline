# src/api/model_integrity.py
# Contre-mesure CVE-2024-37058 + CVE-2024-37065

import hashlib
import json
import os
import sys

HASHES_FILE = "models/hashes.json"

MODELS = {
    "isolation_forest": "models/isolation_forest.pkl",
    "random_forest":    "models/random_forest.pkl",
    "scaler":           "models/scaler.pkl"
}

def compute_hash(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def generate_hashes():
    """Génère les hashs de référence — à faire UNE SEULE FOIS après entraînement."""
    hashes = {}
    for name, path in MODELS.items():
        if os.path.exists(path):
            hashes[name] = {
                "path":   path,
                "sha256": compute_hash(path)
            }
            print(f"Hash {name}: {hashes[name]['sha256'][:16]}...")
    with open(HASHES_FILE, "w") as f:
        json.dump(hashes, f, indent=2)
    print(f"\nHashs de référence sauvegardés ✅")

def verify_integrity() -> bool:
    """
    Vérifie l'intégrité des modèles contre les hashs de référence.
    NE régénère PAS les hashs — compare uniquement.
    """
    if not os.path.exists(HASHES_FILE):
        print("ERREUR: Fichier de hashs introuvable — lancer d'abord: python3 model_integrity.py generate")
        return False

    with open(HASHES_FILE) as f:
        reference = json.load(f)

    all_ok = True
    for name, info in reference.items():
        path = info["path"]
        if not os.path.exists(path):
            print(f"ERREUR: Modèle manquant : {path}")
            all_ok = False
            continue

        current = compute_hash(path)
        expected = info["sha256"]

        if current != expected:
            print(f"ALERTE SÉCURITÉ: Modèle '{name}' modifié !")
            print(f"  Hash attendu : {expected[:32]}...")
            print(f"  Hash actuel  : {current[:32]}...")
            all_ok = False
        else:
            print(f"Intégrité OK : {name} ✅")

    return all_ok

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        print("Génération des hashs de référence...")
        generate_hashes()
    else:
        print("Vérification de l'intégrité des modèles...")
        ok = verify_integrity()
        print(f"\nRésultat : {'✅ Tous les modèles sont intègres' if ok else '❌ COMPROMISSION DÉTECTÉE'}")
        sys.exit(0 if ok else 1)
