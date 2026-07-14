# src/api/model_signing.py
# Signature numérique RSA des modèles ML
# Protection contre CVE-2024-37058 + CVE-2024-37065

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import os, json

KEYS_DIR      = "models/keys"
SIGS_DIR      = "models/signatures"
PRIVATE_KEY   = f"{KEYS_DIR}/private_key.pem"
PUBLIC_KEY    = f"{KEYS_DIR}/public_key.pem"
SIGS_FILE     = f"{SIGS_DIR}/signatures.json"

MODELS = {
    "isolation_forest": "models/isolation_forest.pkl",
    "random_forest":    "models/random_forest.pkl",
    "scaler":           "models/scaler.pkl"
}

def generate_keys():
    """Génère une paire de clés RSA-2048."""
    os.makedirs(KEYS_DIR, exist_ok=True)

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    # Sauvegarder clé privée
    with open(PRIVATE_KEY, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    # Sauvegarder clé publique
    with open(PUBLIC_KEY, "wb") as f:
        f.write(private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))
    print("Paire de clés RSA-2048 générée ✅")
    print(f"  Clé privée : {PRIVATE_KEY} (GARDER SECRÈTE)")
    print(f"  Clé publique : {PUBLIC_KEY} (peut être partagée)")
    return private_key

def sign_models():
    """Signe tous les modèles avec la clé privée."""
    os.makedirs(SIGS_DIR, exist_ok=True)

    # Charger clé privée
    with open(PRIVATE_KEY, "rb") as f:
        private_key = serialization.load_pem_private_key(
            f.read(), password=None
        )

    signatures = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"Modèle manquant : {path}")
            continue

        # Lire le fichier modèle
        with open(path, "rb") as f:
            model_bytes = f.read()

        # Signer avec RSA-PSS + SHA-256
        signature = private_key.sign(
            model_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        signatures[name] = {
            "path":      path,
            "signature": signature.hex()  # Convertir en hex pour JSON
        }
        print(f"Modèle signé : {name} ✅")
        print(f"  Signature : {signature.hex()[:32]}...")

    with open(SIGS_FILE, "w") as f:
        json.dump(signatures, f, indent=2)
    print(f"\nSignatures sauvegardées : {SIGS_FILE} ✅")

def verify_signatures() -> bool:
    """Vérifie les signatures de tous les modèles."""
    if not os.path.exists(PUBLIC_KEY):
        print("ERREUR: Clé publique introuvable")
        return False
    if not os.path.exists(SIGS_FILE):
        print("ERREUR: Fichier de signatures introuvable")
        return False

    # Charger clé publique
    with open(PUBLIC_KEY, "rb") as f:
        public_key = serialization.load_pem_public_key(f.read())

    with open(SIGS_FILE) as f:
        signatures = json.load(f)

    all_ok = True
    for name, info in signatures.items():
        path = info["path"]
        if not os.path.exists(path):
            print(f"ERREUR: Modèle manquant : {path}")
            all_ok = False
            continue

        with open(path, "rb") as f:
            model_bytes = f.read()

        try:
            public_key.verify(
                bytes.fromhex(info["signature"]),
                model_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            print(f"Signature valide : {name} ✅")
        except Exception:
            print(f"ALERTE: Signature invalide pour '{name}' ❌")
            print(f"  Le modèle a été modifié ou remplacé !")
            all_ok = False

    return all_ok

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"

    if cmd == "generate-keys":
        generate_keys()
    elif cmd == "sign":
        sign_models()
    elif cmd == "verify":
        print("Vérification des signatures RSA...")
        ok = verify_signatures()
        print(f"\nRésultat : {'✅ Tous les modèles authentifiés' if ok else '❌ AUTHENTICITÉ COMPROMISE'}")
        sys.exit(0 if ok else 1)
