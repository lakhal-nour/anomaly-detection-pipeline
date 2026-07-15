FROM python:3.12-slim

LABEL maintainer="lakhal-nour"
LABEL project="SIEM-Anomaly-Detection"
LABEL version="2.0.0"

WORKDIR /app

# Utiliser le requirements minimal pour Docker
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copier le code
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8002

# Utilisateur non-root
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8002"]
