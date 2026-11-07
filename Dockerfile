# Single container: builds the React frontend, then serves it + the FastAPI API from uvicorn on port 8000.
#   docker build -t p5-vehicle .
#   docker run --rm -p 8000:8000 p5-vehicle            → http://localhost:8000
# GPU (needs the NVIDIA container toolkit): build with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124
# and run with --gpus all.

# ---------- 1. frontend build ----------
FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---------- 2. backend runtime ----------
FROM python:3.12-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 YOLO_CONFIG_DIR=/tmp/Ultralytics BACKEND=torch
# libGL / glib are needed by the opencv-python wheel that ultralytics pulls in
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --index-url ${TORCH_INDEX} torch torchvision \
 && pip install -r backend/requirements.txt

COPY artifacts/ artifacts/
COPY plots/ plots/
COPY samples/ samples/
COPY backend/ backend/
COPY --from=web /web/dist frontend/dist

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD python -c "import urllib.request,sys; sys.exit(0 if b'\"model_loaded\":true' in urllib.request.urlopen('http://127.0.0.1:8000/api/health').read() else 1)"
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
