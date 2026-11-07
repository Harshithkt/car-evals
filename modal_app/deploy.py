"""Host the whole app (FastAPI API + built React frontend) on Modal, CPU only.

    cd frontend && npm run build && cd ..            # the frontend is served from frontend/dist
    .venv/bin/modal deploy modal_app/deploy.py       # → https://<workspace>--p5-vehicle-app-web.modal.run
    P5_WARM=1 .venv/bin/modal deploy modal_app/deploy.py   # demo day: keep one container warm (~$0.13/h)
    .venv/bin/modal deploy modal_app/deploy.py       # afterwards: back to scale-to-zero
    .venv/bin/modal app stop p5-vehicle-app          # take it offline

Cost controls
  * CPU only: 2 physical cores (hard limit, so billing can't exceed 2 cores) + 4 GiB. About $0.13 per hour while a container runs.
  * Scale to zero: after 5 idle minutes the container stops and billing stops. The next visitor waits for a cold start.
  * max_containers=1: a public link can never fan out into many billed containers.
Both classifier engines are loaded: ONNX INT8 (fast; used by live camera / batch fast mode) and PyTorch FP32 (Grad-CAM).
"""
import os
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
CPUS = 2
WARM = int(os.environ.get('P5_WARM', '0'))

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
    .add_local_dir(ROOT / 'plots', '/app/plots')
    .add_local_dir(ROOT / 'samples', '/app/samples')
    .add_local_dir(ROOT / 'frontend' / 'dist', '/app/frontend/dist')
)

app = modal.App('p5-vehicle-app', image=image)


@app.function(
    cpu=(float(CPUS), float(CPUS)),  # (request, hard limit)
    memory=4096,
    min_containers=WARM,
    max_containers=1,
    scaledown_window=300,
    timeout=120,
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
