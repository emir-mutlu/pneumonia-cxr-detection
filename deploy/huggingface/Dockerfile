# Hugging Face Space (Docker SDK): clinical web demo on CPU.
FROM python:3.11-slim

RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/home/user/app/src:/home/user/app \
    MODEL_DIR=/home/user/app/models \
    OMP_NUM_THREADS=2
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "app.web:app", "--host", "0.0.0.0", "--port", "7860"]
