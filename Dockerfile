# Dockerfile
FROM python:3.12-slim

LABEL maintainer="lakhal-nour"
LABEL project="SIEM-Anomaly-Detection"
LABEL version="1.0.0"

WORKDIR /app

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY src/ ./src/
COPY models/ ./models/

# Port exposé
EXPOSE 8002

# Utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 appuser
USER appuser

# Lancer l'API
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8002"]
