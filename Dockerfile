FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    ANONYMIZED_TELEMETRY=False \
    GEMINI_MODEL=gemini-flash-lite-latest \
    MIN_SIM=0.45 \
    RETRIEVAL_MODE=rerank \
    API_URL=http://127.0.0.1:8000 \
    RATE_LIMIT_PER_MIN=30

WORKDIR /app

# CPU-only torch keeps the image far smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Always take the newest Gemini SDK: older versions reject newer-style API keys with 401 ACCESS_TOKEN_TYPE_UNSUPPORTED.
RUN pip install --no-cache-dir -U google-genai

COPY api.py service_utils.py generate.py retriever.py index.py ui.py start.py ./
COPY data/chunks.jsonl data/chunks.jsonl

# Build the vector index and download both models at build time,
# so the container starts fast and needs no internet except for the Gemini call.
RUN python index.py \
 && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# start.py runs the API (port 8000, internal) and the Streamlit UI (port 7860, public).
# One API worker: every worker would load its own copy of the models into memory.
CMD ["python", "start.py"]