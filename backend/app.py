"""FastAPI backend for the P5 fine-grained vehicle recognition system.

Run from the project root:
    uvicorn backend.app:app --reload

Environment variables:
    BACKEND        torch (default) | onnx   — classifier runtime (Grad-CAM only with torch)
    DEVICE         auto (default) | cuda | cpu
    ARTIFACTS_DIR  default ./artifacts
    PLOTS_DIR      default ./plots
    SAMPLES_DIR    default ./samples
    CORS_ORIGINS   comma-separated, default http://localhost:5173,http://127.0.0.1:5173
    MAX_UPLOAD_MB  default 10
    WARMUP         1 (default) | 0
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import sys
import threading
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
ART = Path(os.environ.get('ARTIFACTS_DIR', ROOT / 'artifacts')).resolve()
PLOTS = Path(os.environ.get('PLOTS_DIR', ROOT / 'plots')).resolve()
SAMPLES = Path(os.environ.get('SAMPLES_DIR', ROOT / 'samples')).resolve()
FRONTEND_DIST = ROOT / 'frontend' / 'dist'
MAX_UPLOAD_BYTES = int(float(os.environ.get('MAX_UPLOAD_MB', '10')) * 1024 * 1024)
MAX_PIXELS = 40_000_000  # reject decompression bombs / absurdly large images
ALLOWED_FORMATS = {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp', 'BMP': 'bmp'}
ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
SAFE_NAME = re.compile(r'^[A-Za-z0-9_\-]{1,64}$')

log = logging.getLogger('p5')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

# inference.py does `from classical_cv import ...`, so the artifacts folder must be importable.
sys.path.insert(0, str(ART))

STATE: dict[str, Any] = {'rec': None, 'error': None, 'warnings': []}
PREDICT_LOCK = threading.Lock()  # VehicleRecognizer keeps Grad-CAM hook state on the instance → serialise calls


# ---------------------------------------------------------------- helpers
def to_jsonable(o: Any) -> Any:
    """Recursively cast numpy scalars/arrays to plain Python types."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def encode_rgb(img: np.ndarray | None, fmt: str = 'JPEG', quality: int = 90, max_side: int | None = None) -> str | None:
    """RGB uint8 numpy → data URL (base64). `max_side` downsizes for display only (keeps responses small over slow links)."""
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    pil = Image.fromarray(arr)
    if max_side and max(pil.size) > max_side:
        pil.thumbnail((max_side, max_side), Image.LANCZOS)
    if fmt == 'JPEG':
        pil.convert('RGB').save(buf, 'JPEG', quality=quality)
    else:
        pil.save(buf, fmt, optimize=True)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def pick_device() -> str:
    import torch
    want = os.environ.get('DEVICE', 'auto').lower()
    if want == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if want == 'cuda' and not torch.cuda.is_available():
        STATE['warnings'].append('DEVICE=cuda requested but CUDA is not available; using CPU.')
        return 'cpu'
    return want


def load_recognizer() -> None:
    from sklearn.exceptions import InconsistentVersionWarning

    from inference import VehicleRecognizer  # artifacts/inference.py

    backend = os.environ.get('BACKEND', 'torch').lower()
    if backend not in ('torch', 'onnx'):
        raise RuntimeError(f"BACKEND must be 'torch' or 'onnx', got {backend!r}")
    has_onnx = (ART / 'classifier_int8.onnx').exists()
    if backend == 'torch' and not (ART / 'classifier_best.pth').exists() and has_onnx:
        # e.g. a fresh git clone: the 112 MB PyTorch checkpoint is not in the repository
        STATE['warnings'].append('artifacts/classifier_best.pth not found; running the ONNX INT8 classifier (no Grad-CAM).')
        backend = 'onnx'
    if backend == 'onnx' and not has_onnx:
        STATE['warnings'].append('BACKEND=onnx requested but artifacts/classifier_int8.onnx is missing; using torch.')
        backend = 'torch'
    device = pick_device()
    # With the PyTorch default, also load the INT8 ONNX classifier (if exported) as a fast secondary engine for
    # live camera / batch requests. Disable with LOAD_ONNX=0. BACKEND=onnx loads only the ONNX engine.
    engines = [backend] + (['onnx'] if backend == 'torch' and has_onnx and os.environ.get('LOAD_ONNX', '1') != '0' else [])

    recs = {}
    for eng in engines:
        t0 = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', InconsistentVersionWarning)
            recs[eng] = VehicleRecognizer(str(ART), device=device, backend=eng)
        sk = {str(w.message).split(' from version ')[1].split(' ')[0] for w in caught
              if issubclass(w.category, InconsistentVersionWarning) and ' from version ' in str(w.message)}
        if sk and not any('scikit-learn' in w for w in STATE['warnings']):
            import sklearn
            STATE['warnings'].append(f'classical_lr.joblib was saved with scikit-learn {", ".join(sorted(sk))}; '
                                     f'running {sklearn.__version__} (verified working by scripts/smoke_test.py).')
        log.info('VehicleRecognizer loaded in %.1fs (engine=%s, device=%s)', time.time() - t0, eng, recs[eng].device)

    if os.environ.get('WARMUP', '1') != '0':
        sample = next(iter(sorted(SAMPLES.glob('*.jpg'))), None)
        img = str(sample) if sample else np.full((480, 640, 3), 127, np.uint8)
        for eng, rec in recs.items():
            t0 = time.time()
            rec.predict(img, max_vehicles=1, return_stages=True, gradcam=eng == 'torch')
            log.info('Warm-up prediction (%s) done in %.1fs', eng, time.time() - t0)

    STATE.update(rec=recs[backend], recs=recs, backend=backend, device=str(recs[backend].device), loaded_at=time.time())


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_recognizer()
    except Exception as e:  # keep the API up so /api/health can report the problem
        log.exception('Failed to load model')
        STATE['error'] = f'{type(e).__name__}: {e}'
    yield


# ---------------------------------------------------------------- app
app = FastAPI(title='P5 Vehicle Recognition API', version='1.0', lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=2048)  # JSON with base64 images compresses ~25%
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get(
        'CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(',') if o.strip()],
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)


@app.get('/api/health')
def health():
    cfg = read_json(ART / 'config.json', {})
    return {
        'status': 'ok' if STATE['rec'] is not None else 'error',
        'model_loaded': STATE['rec'] is not None,
        'device': STATE.get('device'),
        'backend': STATE.get('backend'),
        'engines': list(STATE.get('recs', {})),
        'gradcam_available': 'torch' in STATE.get('recs', {}),
        'error': STATE['error'],
        'warnings': STATE['warnings'],
        'thresholds': {k: cfg.get(k) for k in ('tau', 'tau_make', 'det_conf')},
        'max_upload_mb': MAX_UPLOAD_BYTES / 1024 / 1024,
    }


def _decode_upload(data: bytes, filename: str | None) -> Image.Image:
    if not data:
        raise HTTPException(400, 'The uploaded file is empty.')
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f'File is {len(data) / 1e6:.1f} MB; the limit is {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB.')
    if filename and Path(filename).suffix and Path(filename).suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(415, f'Unsupported file type {Path(filename).suffix!r}. Use JPG, PNG, WEBP or BMP.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            im = Image.open(io.BytesIO(data))
            fmt = im.format
            if fmt not in ALLOWED_FORMATS:
                raise HTTPException(415, f'Unsupported image format {fmt!r}. Use JPG, PNG, WEBP or BMP.')
            if im.width * im.height > MAX_PIXELS:
                raise HTTPException(413, f'Image is {im.width}x{im.height}; please upload something under 40 megapixels.')
            im.load()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombWarning,
            Image.DecompressionBombError) as e:
        raise HTTPException(400, f'Could not read the image ({type(e).__name__}). Is the file a valid, uncorrupted image?')
    im = ImageOps.exif_transpose(im)  # respect phone-camera orientation
    if im.width < 16 or im.height < 16:
        raise HTTPException(400, f'Image is too small ({im.width}x{im.height}).')
    return im.convert('RGB')


@app.post('/api/predict')
def predict(
    file: UploadFile = File(..., description='Image file (jpg/png/webp/bmp, ≤10 MB)'),
    gradcam: bool = Query(True),
    stages: bool = Query(True),
    max_vehicles: int = Query(5, ge=1, le=20),
    engine: str | None = Query(None, pattern='^(torch|onnx)$', description='Classifier engine; default = BACKEND'),
):
    if STATE['rec'] is None:
        raise HTTPException(503, f"Model is not loaded: {STATE['error'] or 'still starting'}")
    engine = engine or STATE['backend']
    rec = STATE['recs'].get(engine)
    if rec is None:
        raise HTTPException(400, f"Engine {engine!r} is not loaded. Available: {', '.join(STATE['recs'])}.")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    img = _decode_upload(data, file.filename)

    want_cam = gradcam and engine == 'torch'
    with PREDICT_LOCK:
        out = rec.predict(img, max_vehicles=max_vehicles, return_stages=stages, gradcam=want_cam)

    t_enc = time.time()
    resp = {
        'filename': file.filename,
        'image_size': [img.width, img.height],
        'vehicles': to_jsonable(out['vehicles']),
        # boxes / image_size stay in original-image pixels; only the returned pictures are display-sized
        'annotated_image': encode_rgb(out['annotated_image'], quality=85, max_side=1280),
        'gradcam': encode_rgb(out.get('gradcam'), quality=85, max_side=640) if want_cam else None,
        'gradcam_note': None if want_cam else (
            'Grad-CAM is not available with the ONNX backend.' if gradcam else None),
        'stages': [{'name': k, 'image_b64': encode_rgb(v, quality=78, max_side=480)} for k, v in out.get('stages', {}).items()]
        if stages else [],
        'classical_info': to_jsonable(out.get('classical_info', {})),
        'timings_ms': to_jsonable(out['timings_ms']),
        'backend': engine,
        'device': STATE['device'],
    }
    resp['timings_ms']['encode'] = (time.time() - t_enc) * 1000
    return resp


@app.get('/api/metrics')
def metrics():
    # metrics.json written by pandas/json may contain NaN (e.g. optimisation rows without latency) → null
    return to_jsonable(read_json(ART / 'metrics.json', {}))


@app.get('/api/classes')
def classes():
    return read_json(ART / 'class_meta.json', [])


@app.get('/api/config')
def config():
    return read_json(ART / 'config.json', {})


@app.get('/api/plots')
def list_plots():
    return sorted(p.stem for p in PLOTS.glob('*.png')) if PLOTS.is_dir() else []


def _safe_file(base: Path, name: str, ext: str) -> Path:
    stem = name[: -len(ext)] if name.lower().endswith(ext) else name
    if not SAFE_NAME.match(stem):
        raise HTTPException(400, 'Invalid name.')
    path = (base / f'{stem}{ext}').resolve()
    if path.parent != base or not path.is_file():
        raise HTTPException(404, f'{stem}{ext} not found.')
    return path


_PLOT_CACHE: dict[tuple[str, int, float], bytes] = {}


@app.get('/api/plots/{name}')
def get_plot(name: str, w: int | None = Query(None, ge=200, le=4000, description='Resize to this width (WebP) for page display')):
    path = _safe_file(PLOTS, name, '.png')
    headers = {'Cache-Control': 'public, max-age=3600'}
    if w is None:
        return FileResponse(path, media_type='image/png', headers=headers)
    key = (path.name, w, path.stat().st_mtime)
    if key not in _PLOT_CACHE:
        with Image.open(path) as im:
            im = im.convert('RGB')
            if im.width > w:
                im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, 'WEBP', quality=85, method=4)
        _PLOT_CACHE[key] = buf.getvalue()
    return Response(_PLOT_CACHE[key], media_type='image/webp', headers=headers)


@app.get('/api/samples')
def list_samples():
    gt = read_json(SAMPLES / 'ground_truth.json', {})
    return [{'name': p.stem, 'url': f'/api/samples/{p.stem}', 'ground_truth': gt.get(p.name)}
            for p in sorted(SAMPLES.glob('*.jpg'))] if SAMPLES.is_dir() else []


@app.get('/api/samples/{name}')
def get_sample(name: str):
    return FileResponse(_safe_file(SAMPLES, name, '.jpg'), media_type='image/jpeg')


# Serve the production frontend build (npm run build) from the same server, if present.
if FRONTEND_DIST.is_dir():
    app.mount('/', StaticFiles(directory=FRONTEND_DIST, html=True), name='frontend')
