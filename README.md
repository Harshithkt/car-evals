<div align="center">

# Fine-Grained Vehicle Recognition

**Identify the make, model and year of a car from a single photo, across 196 classes.**
A hybrid system of classical computer vision, YOLOv8 detection and a ConvNeXt classifier, with calibrated confidence,
low-confidence handling and visual explanations, served as a FastAPI + React web app.

[**Live demo**](https://harshithkt06--p5-vehicle-app-web.us-east.modal.run) ·
[Results](#results) ·
[Quick start](#quick-start) ·
[How it works](#how-it-works) ·
[API](#api) ·
[Deploy](#deploy-on-modal)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ConvNeXt--Tiny-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8s-111F68)
![ONNX](https://img.shields.io/badge/ONNX_Runtime-INT8-005CED?logo=onnx&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React_19-Vite_+_TypeScript-61DAFB?logo=react&logoColor=black)

<img src="docs/screenshots/landing.jpg" alt="Landing page" width="900">

</div>

> The live demo scales to zero when idle, so the first request after a quiet period takes about 15 s while the models load.

---

## Results

All numbers are on the **8,041 unseen Stanford Cars test images**. Thresholds and temperatures were tuned on a separate 10% validation split.

| | Result |
|---|---|
| **Final hybrid system** | **94.7% top-1** · 99.4% top-5 · **98.4% make** · 95.2% make + model |
| Classical vs deep vs hybrid (top-1) | 14.7% → 94.6% → 94.7% |
| Vehicle detection (fine-tuned YOLOv8s) | mAP@0.5 **0.990** · mAP@0.5:0.95 0.934 · mean IoU 0.954 |
| Where the errors are | wrong year only 0.56% · same make, wrong model 3.18% · wrong make 1.59% |
| Calibration (ECE) | 13.4% raw → **1.6%** after temperature scaling (1.7% for the fused hybrid) |
| Decisions at τ = 0.90 / τ_make = 0.30 | **87.1% confident (97.8% correct)** · 12.3% make-only · 0.5% low confidence |
| Robustness (perturbed before detection) | dark 93.8% · very dark 94.0% · over-exposed 93.1% · rotated 15° 92.8% · 25% / 40% occlusion 89.8% / 84.6% · ×0.25 / ×0.1 resolution 91.8% / 58.9% |
| Optimisation | ONNX INT8: **34 MB vs 112 MB (3.3× smaller)**, top-1 94.7% vs 94.6% (1,000 test crops); classification ≈ 70 ms vs ≈ 650 ms on a laptop CPU |

The hybrid's gain over the deep model alone is small (+0.1 pt clean, ≤ +0.5 pt under perturbations). Its value is in the classical
fallbacks, interpretable intermediate results and calibrated decisions, not raw accuracy. The full numbers, ablations and plots are
in the app's *Model comparison*, *Robustness & confidence* and *Failure analysis* pages, and in [`artifacts/metrics.json`](artifacts/metrics.json).

---

## Features

### Recognize
Upload, drag-and-drop or paste a photo, or pick a sample. Every detected vehicle gets **make · model · year**, a calibrated confidence,
a decision status (*confident* / *uncertain model* / *low confidence*), the top-5 alternatives, the detector source (with fallbacks
flagged), a crop of the vehicle and a **Grad-CAM** heat-map. Results export as a printable PDF report or JSON.

<img src="docs/screenshots/recognize.jpg" alt="Recognize page" width="820">

### Explanations on every result
- **Why not the runner-up?** Explains the closest rival (a different year of the same model, a sibling model, or a look-alike from another brand) and the margin.
- **Threshold playground.** Drag τ and τ_make and watch the decision change, using the same rule as the deployed pipeline.

<img src="docs/screenshots/explain.jpg" alt="Runner-up explanation and threshold playground" width="700">

### Live camera
Point a webcam (or a phone, on the hosted HTTPS site) at a car and boxes and labels are drawn over the live video. Live frames use the
fast INT8 classifier.

<img src="docs/screenshots/camera.jpg" alt="Live camera" width="820">

### Stress lab
Degrade a photo in the browser: under/over-exposure, night, low contrast, fog, blur, CCTV low-res, sensor noise, occlusion or camera
tilt. Then compare before vs after, and **sweep severity 0 → 100%** to find where the prediction breaks and whether the system flags
its own uncertainty instead of failing silently.

<img src="docs/screenshots/stress-lab.jpg" alt="Stress lab" width="820">

### Batch analysis
Drop many images and get a queue with progress and ETA, a summary (makes found, decision mix, accuracy against ground truth) and **CSV / JSON export**.

<img src="docs/screenshots/batch.jpg" alt="Batch analysis" width="820">

### Evaluation dashboards
Model comparison (classical vs deep vs hybrid, ablation, detection, training curves, optimisation), robustness and confidence
(12 perturbation conditions, slices, ECE, reliability, risk–coverage), failure analysis (error hierarchy, most-confused pairs,
cause tagging, Grad-CAM gallery), a coverage catalogue of all 196 classes, and a requirement checklist.

<p>
<img src="docs/screenshots/robustness.jpg" alt="Robustness" width="49%">
<img src="docs/screenshots/comparison.jpg" alt="Model comparison" width="49%">
</p>

---

## How it works

```mermaid
flowchart LR
    A[Photo] --> B[Classical preprocessing<br/>z-score · CLAHE · Canny · contours]
    A --> C[YOLOv8s detector<br/>fine-tuned on Stanford Cars]
    C -- no box --> D[Fallbacks<br/>low-conf box → contour ROI → whole image]
    C --> E[Vehicle crop]
    D --> E
    E --> F[Adaptive illumination<br/>normalisation]
    F --> G[ConvNeXt-Tiny<br/>+ flip TTA]
    E --> H[Classical features<br/>HOG · LBP/GLCM · colour · geometry<br/>→ PCA → LogReg]
    G --> I[Late fusion<br/>log softmax z/T + α · log p_classical]
    H --> I
    I --> J[Temperature<br/>calibration T₂]
    J --> K{Decision}
    K -- p ≥ τ --> L[Make · Model · Year]
    K -- p_make ≥ τ_make --> M[Likely make, model uncertain]
    K -- otherwise --> N[Low confidence + top guesses]
```

| Stage | Details |
|---|---|
| **Classical CV** | Aspect-preserving resize, per-channel z-score, adaptive gamma + CLAHE on LAB lightness, auto-threshold Canny, morphological closing, contour analysis and a contour-based vehicle ROI. Features: HOG (shape), LBP + GLCM (texture), HSV/LAB statistics (colour), aspect ratio, solidity, circularity, Hu moments, left–right symmetry (geometry). |
| **Detection** | YOLOv8s pretrained on COCO and fine-tuned on Stanford Cars boxes. If it finds nothing, the pipeline falls back to the best low-confidence box, then the classical contour ROI, then the whole image. |
| **Classification** | ConvNeXt-Tiny (ImageNet-22k pretrained, `timm`) fine-tuned for 20 epochs on padded vehicle crops at 288 px, with flip test-time augmentation. |
| **Hybrid fusion** | Classical features → standardise → PCA(256) → logistic regression, late-fused with the CNN: `log_softmax(z/T) + α·log p_classical`, with α tuned on validation. |
| **Calibration and decisions** | Temperatures T and T₂ fitted on validation. Thresholds τ / τ_make chosen on validation for 99% / 98% accuracy on accepted predictions. |
| **Optimisation** | FP16, ONNX FP32, ONNX dynamic INT8 quantisation and L1 pruning were benchmarked. The app serves FP32 (for Grad-CAM) and INT8 (for speed). |

**Dataset.** [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html): 16,185 images, 196 classes at make / model / year level,
one bounding box per image. It has no number-plate annotations or video, so licence-plate reading and vehicle tracking are out of scope.
Anything outside the 196 classes is mapped to the closest known class, which is why the confidence handling matters.

---

## Quick start

**Requirements:** Python 3.10+ (tested 3.11 and 3.14), Node.js 20.19+, about 2 GB of disk. No GPU needed.

### macOS / Linux
```bash
git clone https://github.com/Harshithkt/car-evals.git && cd car-evals

python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
python scripts/smoke_test.py                 # loads the models, predicts the 6 sample images

uvicorn backend.app:app --reload --reload-dir backend          # API on http://localhost:8000

# second terminal
cd frontend && npm install && npm run dev    # app on http://localhost:5173
```

### Windows (PowerShell)
```powershell
git clone https://github.com/Harshithkt/car-evals.git; cd car-evals

py -m venv .venv; .venv\Scripts\Activate.ps1   # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r backend\requirements.txt
python scripts\smoke_test.py

uvicorn backend.app:app --reload --reload-dir backend

# second terminal
cd frontend; npm install; npm run dev
```

### Model files
The repository includes the YOLOv8 detector, the **INT8 ONNX classifier**, the classical model and all metadata, so the app runs right after cloning.

`artifacts/classifier_best.pth` (the 112 MB PyTorch checkpoint) is over GitHub's 100 MB file limit and is **not in the repository**.
Without it, the backend automatically uses the INT8 classifier, and everything works except Grad-CAM. To enable PyTorch inference
and Grad-CAM, put the checkpoint (exported by the training notebook) in `artifacts/`. It is picked up on the next start.

### Single server (production)
```bash
cd frontend && npm run build && cd ..
uvicorn backend.app:app --host 0.0.0.0 --port 8000      # serves the API and the built app on http://localhost:8000
```

A [`Dockerfile`](Dockerfile) builds the same single-server image (`docker build -t car-evals . && docker run -p 8000:8000 car-evals`).

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `BACKEND` | `torch` | Primary classifier: `torch` (PyTorch FP32, Grad-CAM) or `onnx` (INT8). Falls back to `onnx` if the PyTorch checkpoint is missing. |
| `LOAD_ONNX` | `1` | With `BACKEND=torch`, also load the INT8 classifier as a fast second engine. |
| `DEVICE` | `auto` | `auto` (CUDA if available), `cuda` or `cpu` |
| `MAX_UPLOAD_MB` | `10` | Upload size limit |
| `CORS_ORIGINS` | `http://localhost:5173,…` | Allowed browser origins |
| `WARMUP` | `1` | Run a warm-up prediction at startup |
| `ARTIFACTS_DIR`, `PLOTS_DIR`, `SAMPLES_DIR` | `./artifacts`, `./plots`, `./samples` | Data locations |

## API

Interactive docs at `/docs` (Swagger UI).

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict?engine=torch&gradcam=true&stages=true&max_vehicles=5` | Multipart `file` (JPG / PNG / WEBP / BMP, ≤ 10 MB). Returns `vehicles[]` (make, model, year, confidence, status, message, top-5, box, detector source), `annotated_image`, `gradcam`, `stages[]`, `classical_info`, `timings_ms`. Images are base64 data URLs. |
| `GET` | `/api/health` | Model status, device, engines, thresholds |
| `GET` | `/api/metrics` | Every evaluation result |
| `GET` | `/api/classes` | The 196 classes (`id`, `name`, `make`, `model`, `year`) |
| `GET` | `/api/plots`, `/api/plots/{name}?w=1400` | Evaluation plots (full PNG, or resized WebP with `w`) |
| `GET` | `/api/samples`, `/api/samples/{name}` | Sample test images with ground truth |

```bash
curl -F file=@samples/test_00500.jpg "http://localhost:8000/api/predict?stages=false&gradcam=false"
```

---

## Deploy on Modal

[`modal_app/deploy.py`](modal_app/deploy.py) runs the whole app (API + frontend) in one **CPU-only** container on [Modal](https://modal.com):
2 cores (hard limit) + 4 GiB, **scale-to-zero** after 5 idle minutes, and at most one container. That's about $0.13 per hour while running and nothing when idle.

```bash
pip install modal && modal token new
cd frontend && npm run build && cd ..
modal deploy modal_app/deploy.py                  # → https://<workspace>--p5-vehicle-app-web.<region>.modal.run
P5_WARM=1 modal deploy modal_app/deploy.py        # keep one container warm (e.g. during a demo)
modal app stop p5-vehicle-app
```

Deployment uses the model files in your local `artifacts/`, so include `classifier_best.pth` if you want Grad-CAM online.

---

## Reproducing the evaluation

| Notebook | Purpose |
|---|---|
| [`notebook/P5_Vehicle_Recognition.ipynb`](notebook/P5_Vehicle_Recognition.ipynb) | Data preparation, classical CV, YOLOv8 fine-tuning, ConvNeXt training, hybrid fusion, ablation (Colab, T4) |
| [`notebook/P5_Evaluation_Remaining.ipynb`](notebook/P5_Evaluation_Remaining.ipynb) | Loads the trained models (no training) and computes similar-class analysis, robustness, calibration, thresholds, optimisation and failure analysis |

The evaluation notebook can also run on a Modal GPU with `modal run modal_eval/run_eval.py`, which uploads the models, executes the
notebook and downloads the results. Import its output zips with:

```bash
python scripts/import_colab_results.py            # newest P5_artifacts*.zip / P5_plots*.zip in ~/Downloads (--dry-run to preview)
```

The import backs up the current files, rejects unsafe zip paths, refuses to replace a *different* `inference.py` / `classical_cv.py`,
and reports anything still missing.

---

## Project structure

```
├── artifacts/              # models, config.json (thresholds), class_meta.json, metrics.json, inference pipeline
│   ├── inference.py        #   VehicleRecognizer: detection → classification → fusion → decision → Grad-CAM
│   └── classical_cv.py     #   classical preprocessing, edges, contours, ROI, HOG/LBP/GLCM/colour/geometry features
├── backend/app.py          # FastAPI: upload validation, two engines, base64 images, metrics/plots/samples, static frontend
├── frontend/src/           # React 19 + Vite + TypeScript + Tailwind + Recharts
│   ├── pages/              #   Landing, Recognize, Batch, StressLab, Pipeline, Comparison, Robustness, Failures, Catalogue, About
│   └── components/         #   VehicleCard, CameraPanel, charts, tables, UI kit
├── plots/                  # evaluation figures from the notebooks
├── samples/                # 6 unseen test images + ground truth
├── scripts/                # smoke_test.py, build_config.py, import_colab_results.py
├── modal_app/deploy.py     # hosting on Modal (CPU, scale-to-zero)
├── modal_eval/run_eval.py  # run the evaluation notebook on a Modal GPU
├── notebook/               # training + evaluation notebooks
└── Dockerfile
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Header shows "Backend offline" | Start `uvicorn backend.app:app` from the project root, then click retry. |
| No Grad-CAM / "engine onnx" only | `artifacts/classifier_best.pth` is missing; see [Model files](#model-files). |
| `InconsistentVersionWarning` for `classical_lr.joblib` | Harmless (it was saved with scikit-learn 1.6.1). For no warning, use Python ≤ 3.13 and `pip install scikit-learn==1.6.1`. |
| Error loading `yolo_car_best.pt` | `pip install -U ultralytics torch` (verified with torch 2.8–2.15 and ultralytics 8.4). |
| Port 8000 or 5173 in use | `uvicorn backend.app:app --port 8010` and `API_PROXY=http://127.0.0.1:8010 npm run dev`. |
| Camera doesn't start | Browsers allow the camera only on `localhost` or HTTPS. Use the hosted site or a USB webcam. |

## Acknowledgements

- **Stanford Cars**: J. Krause, M. Stark, J. Deng, L. Fei-Fei. *3D Object Representations for Fine-Grained Categorization.* 4th IEEE Workshop on 3D Representation and Recognition (3dRR-13), ICCV 2013. Images are from the [jhpohovey/StanfordCars-Dataset](https://github.com/jhpohovey/StanfordCars-Dataset) mirror.
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [timm](https://github.com/huggingface/pytorch-image-models) (ConvNeXt), [ONNX Runtime](https://onnxruntime.ai), OpenCV, scikit-image, scikit-learn, FastAPI, React, Recharts.
