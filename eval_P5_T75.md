1RUA24CSE0156,1RUA24CSE0164,1RUA24CSE0205

# Evaluation Results: P5, Team T75

**Problem:** P5, Intelligent Fine-Grained Vehicle Recognition and Analysis System
**Team:** T75 (Harshith K T, Himavanth Gowda R S, Kishan Yadav C)
**Repository:** https://github.com/Harshithkt/car-evals (code evaluated at commit `91c48ec`)
**Live application:** https://harshithkt06--p5-vehicle-app-web.us-east.modal.run

## Evaluation protocol

| Item | Value |
|---|---|
| Dataset | Stanford Cars: 196 classes, 49 makes, 189 make + model groups, model years 1991 to 2012 |
| Training split | 7,329 images (official train set minus validation) |
| Validation split | 815 images (10% of the official train set, stratified by class, seed 42). Used for model selection, temperatures T / T₂, fusion weight α and thresholds τ / τ_make |
| **Test split (unseen)** | **8,041 images (official test set)**. Never used for training, tuning or threshold selection |
| Final system | Fine-tuned YOLOv8s finds the vehicle, then adaptive illumination normalisation, ConvNeXt-Tiny with flip TTA, late fusion with the classical LogReg (α = 0.05), temperature calibration (T₂) and a 3-tier decision (τ = 0.90, τ_make = 0.30) |
| Source of numbers | `artifacts/metrics.json`, produced by `notebook/P5_Vehicle_Recognition.ipynb` and `notebook/P5_Evaluation_Remaining.ipynb` (Colab, T4 GPU). CPU latencies were measured on an Apple M5 laptop with the 6 sample images, 3 runs each (18 runs, median). We repeated this on 7 Nov 2026 and got the same numbers within 1% |

---

## Summary (in the order of the problem statement's evaluation metrics)

| # | Evaluation parameter | Metric | Result (final hybrid system, 8,041 unseen test images) |
|---|---|---|---|
| 1 | Classification performance | **Top-1 accuracy** | **94.66%** |
| 1 | Classification performance | **Top-5 accuracy** | **99.35%** |
| 1 | Classification performance | Make accuracy / Make + Model accuracy | 98.41% / 95.22% |
| 2 | Object localization | **Mean IoU** (top detection vs ground-truth box) | **0.954** |
| 2 | Object localization | Localisation accuracy (IoU ≥ 0.5) | 99.88% |
| 3 | Object detection | **mAP@0.5** | **0.990** (own evaluator); 0.995 (Ultralytics `val`) |
| 3 | Object detection | **mAP@0.5:0.95** | **0.934** (own evaluator); 0.937 (Ultralytics `val`) |
| 3 | Object detection | Precision / Recall (Ultralytics) | 0.996 / 0.998 |
| 4 | Model efficiency | **Model size** | 58.7 MB for the deployed fast path (YOLOv8s 22.5 MB + ConvNeXt INT8 34.0 MB + classical 2.3 MB); 136.7 MB with the FP32 classifier |
| 4 | Model efficiency | Classifier compression | ONNX INT8 34.0 MB vs FP32 111.9 MB (**3.3× smaller**), top-1 94.7% vs 94.6% |
| 4 | Model efficiency | **Inference time** (GPU, T4) | Detection 12.8 ms/img; hybrid end-to-end 87.3 ms/img |
| 4 | Model efficiency | **Inference time** (CPU, Apple M5, full pipeline) | **105 ms/img with INT8** (detection 38 + classification 67) vs 677 ms/img with FP32 (detection 38 + classification 639) |
| 5 | Robustness | Top-1 under perturbations (1,000 unseen images, perturbed **before** detection) | clean 94.7%; dark 93.8%; very dark 94.0%; over-exposed 93.1%; low contrast 94.6%; rotated 15° 92.8%; motion blur 91.0%; noise 94.2%; occlusion 25% / 40% 89.8% / 84.6%; resolution ×0.25 / ×0.1 91.8% / 58.9% |
| 6 | Confidence | Calibration (ECE, 15 bins) | 13.36% raw, 1.60% after temperature scaling, **1.73%** for the final hybrid |
| 6 | Confidence | Threshold analysis (τ = 0.90, τ_make = 0.30) | **87.1% of images "confident", 97.8% of those correct**; 12.3% "make only" (make correct 92.3%); 0.5% "low confidence" (34.1% correct) |

---

## 1. Classification performance (Top-1 / Top-5)

| Approach | Top-1 | Top-5 | Make acc. | Make + Model acc. |
|---|---:|---:|---:|---:|
| Classical features (HOG + colour + texture + geometry), PCA, then Logistic Regression, on the YOLO ROI | 14.74% | 35.59% | 24.26% | 15.18% |
| Classical features with an RBF-SVM, on the YOLO ROI | 20.89% | 40.63% | 31.99% | 21.37% |
| Deep learning only (YOLO ROI + ConvNeXt-Tiny) | 94.63% | 99.30% | 98.30% | 95.17% |
| **Hybrid (final system)** | **94.66%** | **99.35%** | **98.41%** | **95.22%** |
| *Reference: deep model on ground-truth crops* | *94.55%* | *99.33%* | *98.33%* | *95.14%* |

Error breakdown of the final system on all 8,041 test images (each error counted once):

| Outcome | Share of test set |
|---|---:|
| Correct (make + model + year) | 94.66% |
| Wrong **year only** (right make + model) | 0.56% |
| Same make, wrong model | 3.18% |
| Wrong make | 1.59% |

Year accuracy given the correct make + model, over the 7 make + model groups that appear in more than one year: 92.36%.

## 2. Object localization (IoU)

Measured on all 8,041 test images. The highest-scoring box is compared with the single ground-truth box.

| Localiser | Mean IoU | Loc. acc. (IoU ≥ 0.5) | Detection rate | Time / image |
|---|---:|---:|---:|---:|
| Classical contour ROI (CLAHE, auto-Canny, closing, then the best-scoring contour) | 0.629 | 75.31% | 100% | 27.3 ms (CPU) |
| YOLOv8s, COCO-pretrained (car / bus / truck), zero-shot | 0.870 | 91.36% | 99.86% | 14.2 ms (T4) |
| **YOLOv8s fine-tuned on Stanford Cars (deployed)** | **0.954** | **99.88%** | **100%** | **12.8 ms (T4)** |

## 3. Object detection (mAP)

| Detector | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---:|---:|---:|---:|
| Classical contour ROI | 0.568 | 0.179 | n/a | n/a |
| YOLOv8s COCO-pretrained (zero-shot) | 0.858 | 0.785 | n/a | n/a |
| **YOLOv8s fine-tuned (own 101-point AP evaluator)** | **0.990** | **0.934** | n/a | n/a |
| **YOLOv8s fine-tuned (Ultralytics `val`, test split)** | **0.995** | **0.937** | **0.996** | **0.998** |

## 4. Model efficiency (size and inference time)

### 4a. Per-approach comparison (Colab T4 GPU for the networks, CPU for the classical features)

| Approach | Model size | Time / image |
|---|---:|---:|
| Classical (YOLO ROI + hand-crafted features + LogReg) | 24.8 MB | 41.1 ms |
| Deep learning (YOLO + ConvNeXt) | 134.5 MB | 34.8 ms |
| Hybrid (YOLO + illumination norm. + ConvNeXt flip TTA + classical fusion) | 136.7 MB | 87.3 ms |

### 4b. Model optimisation (classifier only, 1,000 unseen test crops, batch 1)

| Variant | Technique | Size | Latency | Top-1 |
|---|---|---:|---:|---:|
| PyTorch FP32 (T4 GPU) | baseline | 111.9 MB | 9.7 ms | 94.6% |
| PyTorch FP16 (T4 GPU) | reduced precision | 56.0 MB | 10.4 ms | 94.6% |
| PyTorch FP32 (Colab CPU) | baseline | 111.9 MB | 203.9 ms | 94.6% |
| ONNX Runtime FP32 (Colab CPU) | graph export | 112.0 MB | 226.8 ms | 94.6% |
| **ONNX Runtime INT8 (Colab CPU)** | **dynamic INT8 quantisation** | **34.0 MB** | **158.3 ms** | **94.7%** |
| L1 global pruning 30%, no fine-tune | pruning | 78.4 MB* | n/a | 91.8% |
| L1 global pruning 50%, no fine-tune | pruning | 56.0 MB* | n/a | 16.5% |
| YOLOv8s detector FP32 (T4) | baseline | 22.5 MB | 11.5 ms | n/a |
| YOLOv8s detector FP16 (T4) | reduced precision | 11.3 MB | 12.3 ms | n/a |

\* Size if the zeroed weights were stored sparsely; the dense file stays 111.9 MB.

### 4c. Deployed pipeline on CPU (Apple M5, 6 sample images × 3 runs, median, 1 vehicle per image, flip TTA + fusion)

| Classifier engine | Detection | Classification | Total | Total incl. 11 intermediate stage images |
|---|---:|---:|---:|---:|
| PyTorch FP32 | 38 ms | 639 ms | 677 ms | 698 ms |
| **ONNX INT8** | **38 ms** | **67 ms** | **105 ms** | **119 ms** |
| Classical only (contour ROI + features + LogReg) | n/a | n/a | 7 ms | n/a |

A second run on 7 Nov gave totals of 675 ms (FP32) and 106 ms (INT8) without the stage images, and 691 ms and 120 ms with them. Medians are taken for each part separately, so the parts do not always add up to the total exactly. INT8 is about 6.4 times faster than FP32 end to end.

Live hosted app (Modal, 2 CPU cores), measured from our laptop on 7 Nov 2026 with a 786x492 sample image:

| Request | Time on the server | Time seen by the client | Data returned |
|---|---:|---:|---:|
| Full result (PyTorch, Grad-CAM, 11 stage images) | about 1.4 s | 4.5 to 9.0 s (3 runs) | about 450 KB |
| Fast path (INT8, no stage images) | about 0.5 s | 2.7 to 6.2 s (3 runs) | about 90 KB |
| First request after the app was idle (cold start) | not measured | about 12.6 s | small |

Most of the client time is network distance to the US-east region, so it changes a lot from run to run. The app scales to zero when idle, so the first visitor waits for the models to load.

## 5. Robustness (performance under varying conditions)

1,000 random unseen test images (seed 42). Each perturbation is applied to the **whole image before detection**, so the full pipeline is re-run each time.

| Condition | Group | Mean IoU | Loc. acc. | Deep top-1 | Hybrid top-1 | Hybrid + illum. norm. top-1 | Top-5 | Make acc. | Δ vs clean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | baseline | 0.955 | 100.0% | 94.6% | 94.7% | 94.7% | 99.8% | 99.3% | 0.0 |
| Dark (×0.3) | illumination | 0.951 | 99.8% | 94.3% | 94.3% | 93.8% | 99.6% | 98.7% | −0.9 |
| Very dark (γ = 3) | illumination | 0.942 | 99.7% | 93.8% | 93.5% | 94.0% | 99.4% | 98.7% | −0.7 |
| Over-exposed (×1.6 + 60) | illumination | 0.946 | 99.9% | 93.2% | 93.4% | 93.1% | 99.3% | 97.8% | −1.6 |
| Low contrast (×0.35) | illumination | 0.951 | 99.9% | 94.6% | 94.6% | 94.6% | 99.7% | 98.9% | −0.1 |
| Rotation 15° | viewing angle | 0.677† | 98.8% | 93.1% | 92.9% | 92.8% | 99.4% | 97.9% | −1.9 |
| Occlusion 25% of box | occlusion | 0.924 | 99.2% | 89.9% | 89.8% | 89.8% | 98.9% | 96.7% | −4.9 |
| Occlusion 40% of box | occlusion | 0.885 | 95.8% | 85.3% | 84.8% | 84.6% | 96.7% | 92.4% | −10.1 |
| Resolution ×0.25 | resolution | 0.945 | 99.8% | 91.9% | 91.8% | 91.8% | 98.5% | 97.0% | −2.9 |
| Resolution ×0.1 | resolution | 0.850 | 93.0% | 59.7% | 59.3% | 58.9% | 78.3% | 66.7% | −35.8 |
| Motion blur (15 px) | image quality | 0.922 | 99.4% | 91.0% | 91.0% | 91.0% | 98.7% | 96.1% | −3.7 |
| Gaussian noise (σ = 25) | image quality | 0.942 | 99.7% | 94.5% | 94.3% | 94.2% | 99.5% | 98.7% | −0.5 |

† The rotated ground-truth box is the axis-aligned hull of the rotated box, which is looser than the true box. The IoU therefore understates localisation quality; localisation accuracy at IoU ≥ 0.5 is still 98.8%.

Slices of the clean test set (all 8,041 images):

| Slice | Bucket | n | Top-1 | IoU |
|---|---|---:|---:|---:|
| Viewing angle (box aspect ratio) | front / rear (< 1.3) | 454 | 92.5% | 0.937 |
| | three-quarter (1.3 to 1.8) | 3,104 | 95.5% | 0.954 |
| | side (> 1.8) | 4,483 | 94.3% | 0.956 |
| Vehicle resolution (√box area) | smallest 25% | 2,011 | 94.6% | 0.946 |
| | largest 25% | 2,010 | 95.1% | 0.956 |
| Brightness (HSV-V of vehicle) | darkest 25% | 2,011 | 94.6% | 0.952 |
| | brightest 25% | 2,010 | 94.7% | 0.954 |

## 6. Confidence and threshold analysis

Calibration (Expected Calibration Error, 15 bins, test set):

| Model | ECE |
|---|---:|
| Deep, raw softmax | 13.36% |
| Deep + temperature scaling (T = 0.695) | 1.60% |
| **Hybrid final (fusion + T₂ = 1.034)** | **1.73%** |

Thresholds were chosen **on validation**: τ is the lowest threshold of at least 0.30 that gives at least 99% accuracy on accepted predictions, and τ_make the lowest that gives at least 98% make accuracy. They were then applied unchanged to test. Result: τ = 0.90, τ_make = 0.30.

| Decision tier | Rule | Share of test | Model acc. | Make acc. |
|---|---|---:|---:|---:|
| Confident: show make, model, year | p_top1 ≥ 0.90 | 87.12% | 97.83% | 99.70% |
| Uncertain model: show make only plus candidates | p_top1 < 0.90 and p_make ≥ 0.30 | 12.34% | 75.00% | 92.34% |
| Low confidence: ask for a better image | otherwise | 0.55% | 34.09% | 34.09% |

Coverage and accuracy on the test set when we only keep predictions above a threshold:

| Threshold | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Coverage | 100.0% | 99.93% | 99.60% | 99.03% | 98.25% | 97.16% | 95.76% | 93.99% | 91.69% | 87.12% |
| Accuracy | 94.66% | 94.71% | 94.91% | 95.18% | 95.58% | 95.99% | 96.51% | 96.82% | 97.33% | 97.83% |

The validation target was 99% accuracy for confident predictions; on test it reached 97.83%. The 815-image validation set is too small to estimate a 99% operating point exactly.

---

## 7. Comparative evaluation: classical vs deep learning vs hybrid

| Approach | Top-1 | Top-5 | Make | Make + Model | Size | Time / img (T4) |
|---|---:|---:|---:|---:|---:|---:|
| Classical | 14.74% | 35.59% | 24.26% | 15.18% | 24.8 MB | 41.1 ms |
| Deep learning | 94.63% | 99.30% | 98.30% | 95.17% | 134.5 MB | 34.8 ms |
| **Hybrid** | **94.66%** | **99.35%** | **98.41%** | **95.22%** | 136.7 MB | 87.3 ms |

The hybrid is ahead of deep-only on every accuracy number, but only by a little: +0.04 points top-1 (3 images out of 8,041), +0.05 points top-5 and +0.11 points make accuracy. That is small enough to be noise, so we do not claim it as a real accuracy gain. The hybrid helps in other ways:
- The classical contour ROI takes over when the detector finds nothing.
- Illumination normalisation helps very dark images a little (+0.5 points over the hybrid without it).
- The intermediate images let a person see what each step did.
- Calibrated confidence lets the system say "make only" or "low confidence" instead of guessing the model and year.

## 8. Ablation study (8,041 unseen test images; each row adds or removes one thing)

| # | Configuration | Top-1 | Top-5 | Make | Make + Model | Change in top-1 vs final (points) |
|---|---|---:|---:|---:|---:|---:|
| 1 | Classical features only (LogReg), YOLO ROI | 14.74% | 35.59% | 24.26% | 15.18% | −79.93 |
| 2 | Classical features only (RBF-SVM), YOLO ROI | 20.89% | 40.63% | 31.99% | 21.37% | −73.77 |
| 3 | Deep: whole image (no detection) | 91.42% | 98.05% | 95.83% | 91.98% | −3.25 |
| 4 | Deep: classical contour ROI | 92.54% | 98.35% | 96.68% | 93.07% | −2.13 |
| 5 | Deep: YOLO ROI | 94.63% | 99.30% | 98.30% | 95.17% | −0.04 |
| 6 | Deep: YOLO ROI + flip TTA | 94.68% | 99.35% | 98.36% | 95.22% | +0.01 |
| 7 | Deep: YOLO ROI + TTA + illumination norm. | 94.66% | 99.35% | 98.35% | 95.21% | 0.00 |
| 8 | **Hybrid: row 7 + classical feature fusion (final)** | **94.66%** | **99.35%** | **98.41%** | **95.22%** | **0.00** |
| 9 | Hybrid without illumination norm. | 94.68% | 99.35% | 98.42% | 95.24% | +0.01 |
| 10 | Hybrid without TTA | 94.60% | 99.25% | 98.35% | 95.16% | −0.06 |
| 11 | Upper-bound reference: deep on ground-truth crops | 94.55% | 99.33% | 98.33% | 95.14% | −0.11 |

What each part adds:
- **Finding the car first matters most:** +3.21 points over using the whole image (row 3 to row 5). The classical contour ROI gets about a third of that on its own (+1.12 points, row 3 to row 4).
- **Fine-tuned YOLO crops are as good as the ground-truth crops** (row 5 vs row 11). The YOLO boxes are a bit tighter and more consistent.
- **Flip TTA:** +0.05 points top-1 and +0.05 points top-5 (row 5 to row 6). Taking TTA out of the hybrid costs 0.06 points (row 10).
- **Classical fusion:** +0.06 points make accuracy (row 7 to row 8) and the same top-1. The fusion weight α = 0.05 was picked on the validation set by trying 0 to 1 in steps of 0.05.
- **Illumination normalisation:** no gain on the clean test set (about 0.01 points worse, rows 6 and 7, rows 9 and 8). It only helps on very dark images, by 0.5 points in the robustness test. On moderately dark images (dark ×0.3) it made things 0.5 points worse, so the result is mixed.

## 9. Failure-case analysis

All 429 errors of the final system on the 8,041 test images were each tagged with one cause. Rules are applied in order; see `notebook/P5_Evaluation_Remaining.ipynb`, section E.

| Cause | Errors | Share of errors |
|---|---:|---:|
| Same make, sibling model (e.g. Audi S5 mixed up with A5) | 219 | 51.0% |
| Cross-make look-alike or rebadged twin (e.g. Chevrolet Express mixed up with GMC Savana) | 98 | 22.8% |
| Truncation / occlusion by the image border | 39 | 9.1% |
| Year confusion (same make + model) | 37 | 8.6% |
| Low resolution (vehicle short side < 100 px) | 28 | 6.5% |
| Low light (crop brightness < 70/255) | 6 | 1.4% |
| Localisation error (detector IoU < 0.5) | 2 | 0.5% |

Most confused pairs (true class, predicted class, count):

| True class | Predicted class | Count | Relation |
|---|---|---:|---|
| Dodge Caliber Wagon 2012 | Dodge Caliber Wagon 2007 | 13 | year |
| Chevrolet Silverado 1500 Regular Cab 2012 | Chevrolet Silverado 2500HD Regular Cab 2012 | 12 | same make |
| Audi S5 Coupe 2012 | Audi A5 Coupe 2012 | 12 | same make |
| BMW 6 Series Convertible 2007 | BMW M6 Convertible 2010 | 10 | same make |
| Dodge Sprinter Cargo Van 2009 | Mercedes-Benz Sprinter Van 2012 | 10 | different make |
| Bentley Continental GT Coupe 2012 | Bentley Continental GT Coupe 2007 | 10 | year |
| Audi TTS Coupe 2012 | Audi TT Hatchback 2011 | 9 | same make |
| Chevrolet Express Cargo Van 2007 | Chevrolet Express Van 2007 | 9 | same make |

Hardest classes (test accuracy): Chevrolet Express Cargo Van 2007 55.2%; Chevrolet Express Van 2007 57.1%; Dodge Caliber Wagon 2012 67.5%; Audi 100 Sedan 1994 67.5%; Chevrolet Silverado 1500 Regular Cab 2012 70.5%.

Three failure cases, explained (the Grad-CAM pictures are in `plots/failure_gallery.png` and the longer write-up is in `DESIGN_REPORT_P5_T75.md`, section 4):
1. **Model-year facelifts.** Bentley Continental GT Coupe 2012 was predicted as 2007 with 99.8% confidence. The two generations share the same body and differ only in small lamp and bumper details. A 288 pixel crop barely shows them. The same thing happens with the Dodge Caliber Wagon 2012 and 2007 (13 test images).
2. **Rebadged twins across makes.** Chevrolet Express Cargo Van was predicted as GMC Savana (99.8%), and Dodge Sprinter as Mercedes-Benz Sprinter. These are the same physical vehicle with a different badge. In the gallery example (`06675.jpg`), the image labelled "Chevrolet Express" has a GMC badge on the grille, so some of these "errors" are really wrong labels in the dataset.
3. **Extreme low resolution (×0.1).** Top-1 falls to 58.9% and make accuracy to 66.7%, because the badge, grille and lamp details that tell classes apart are gone. Training only shrank images to ×0.2, so ×0.1 was never seen. The detector still finds the car 93% of the time.

---

The numbers in this file come from `artifacts/metrics.json` and the two notebooks. This file was drafted with help from an AI assistant (Claude). Every number was checked against `metrics.json`, and the CPU and live-app timings were measured again on 7 Nov 2026.
