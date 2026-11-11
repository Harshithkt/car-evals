"""Host the whole app (FastAPI API + built React frontend) on Modal, CPU only.

    cd frontend && npm run build && cd ..            # the frontend is served from frontend/dist
    .venv/bin/modal deploy modal_app/deploy.py       # → https://<workspace>--p5-vehicle-app-web.modal.run
    P5_WARM=1 .venv/bin/modal deploy modal_app/deploy.py   # demo day: keep one container warm (~$0.14/h)
    .venv/bin/modal deploy modal_app/deploy.py       # afterwards: back to scale-to-zero
    .venv/bin/modal app stop p5-vehicle-app          # take it offline

Cost controls
  * CPU only: 2 physical cores (hard limit, so billing can't exceed 2 cores) + 6 GiB. About $0.14 per hour while a container runs.
  * Scale to zero: after 5 idle minutes the container stops and billing stops. The next visitor waits for a cold start.
  * max_containers=1: a public link can never fan out into many billed containers.
Two models are served (all cars 1948-2026 by default, and the Stanford Cars benchmark model). Each has two classifier
engines: ONNX INT8 (fast; used by live camera / batch fast mode) and PyTorch FP32 (Grad-CAM). The default model loads
before the first request is served; the other one loads in the background right after.

Optional LLM second opinion: put NEBIUS_API_KEY=... in the git-ignored .env file (or export it) before `modal deploy`.
It is passed to the container as a Modal secret; without it the button is simply hidden.
"""
import os
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
CPUS = 2
WARM = int(os.environ.get('P5_WARM', '0'))


def _llm_secrets() -> list:
    """Optional LLM second opinion: if NEBIUS_API_KEY is set (environment or the git-ignored .env file), pass it to the
    container as a Modal secret. The key is never baked into the image or printed."""
    env = {}
    p = ROOT / '.env'
    if p.is_file():
        for line in p.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    vals = {k: os.environ.get(k) or env.get(k) for k in ('NEBIUS_API_KEY', 'NEBIUS_VLM_MODEL', 'LLM_MAX_PER_HOUR')}
    vals = {k: v for k, v in vals.items() if v}
    return [modal.Secret.from_dict(vals)] if vals.get('NEBIUS_API_KEY') else []

image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('libgl1', 'libglib2.0-0')
    # CPU-only PyTorch wheels: much smaller image, faster cold starts
    .pip_install('torch==2.8.0', 'torchvision==0.23.0', index_url='https://download.pytorch.org/whl/cpu')
    .pip_install(
        'ultralytics', 'timm', 'onnxruntime', 'opencv-python-headless', 'scikit-image',
        'scikit-learn==1.6.1',  # the version that pickled classical_lr.joblib (no version warning)
        'scipy', 'joblib', 'numpy', 'Pillow', 'fastapi', 'python-multipart',
    )
    .env({
        'BACKEND': 'torch', 'LOAD_ONNX': '1', 'DEVICE': 'cpu', 'WARMUP': '1',
        'OMP_NUM_THREADS': str(CPUS), 'MKL_NUM_THREADS': str(CPUS),
        'YOLO_CONFIG_DIR': '/tmp/Ultralytics', 'PYTHONUNBUFFERED': '1',
    })
    .add_local_dir(ROOT / 'backend', '/app/backend', ignore=['__pycache__', '*.pyc'])
    # only what inference needs: skip the FP32 ONNX / FP16 exports, backups and caches (~170 MB instead of ~330 MB)
    .add_local_dir(ROOT / 'artifacts', '/app/artifacts',
                   ignore=['classifier_fp32.onnx', 'classifier_fp16.pth', 'backup-*', '__pycache__', '*.colab.*', 'requirements.txt'])
    # the all-cars model (default in the app); same file layout, no classical fusion file
    .add_local_dir(ROOT / 'artifacts_modern', '/app/artifacts_modern', ignore=['__pycache__', '*.onnx.data'])
    .add_local_dir(ROOT / 'plots', '/app/plots')
    .add_local_dir(ROOT / 'samples', '/app/samples')
    .add_local_dir(ROOT / 'frontend' / 'dist', '/app/frontend/dist')
)

app = modal.App('p5-vehicle-app', image=image)


@app.function(
    cpu=(float(CPUS), float(CPUS)),  # (request, hard limit)
    memory=6144,  # two models (about 1.6 GB resident) plus headroom for Grad-CAM and large uploads
    min_containers=WARM,
    max_containers=1,
    scaledown_window=300,
    timeout=120,
    secrets=_llm_secrets(),
)
@modal.concurrent(max_inputs=8)  # static files and health checks don't queue behind a prediction
@modal.asgi_app()
def web():
    import sys
    sys.path.insert(0, '/app')

    # Thread pools sized to the 2 reserved cores. Containers can report the host's CPU count, which would
    # oversubscribe the cores and slow everything down. This changes speed only, never the predictions.
    import torch
    torch.set_num_threads(CPUS)
    import onnxruntime as ort
    _Session = ort.InferenceSession

    class _CappedSession(_Session):
        def __init__(self, path, sess_options=None, *args, **kwargs):
            if sess_options is None:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = CPUS
                sess_options.inter_op_num_threads = 1
            super().__init__(path, sess_options, *args, **kwargs)

    ort.InferenceSession = _CappedSession

    from backend.app import app as fastapi_app  # models load in its lifespan startup (with a warm-up call)
    return fastapi_app
