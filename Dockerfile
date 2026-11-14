# Torch-free inference image: ONNX Runtime + FastAPI (~350 MB).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    MODEL_DIR=/app/models

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/pneumonia_cxr/__init__.py src/pneumonia_cxr/inference.py src/pneumonia_cxr/
COPY src/pneumonia_cxr/data/__init__.py src/pneumonia_cxr/data/imaging.py src/pneumonia_cxr/data/
COPY app/__init__.py app/api.py app/
COPY models/model.onnx models/model_meta.json models/

RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
