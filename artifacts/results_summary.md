| Requirement | Result |
|---|---|
| 1. Similar makes/models/years | top-1 94.7%, make 98.4%; errors: year-only 0.56%, same-make 3.18%, other make 1.59% |
| 2. Illumination & viewing angle | dark 93.8%, very dark 94.0%, overexposed 93.1%, rotated 92.8% (clean 94.7%) |
| 3. Occlusion & resolution | occl. 25% 89.8%, 40% 84.6%, ×0.25 res 91.8%, ×0.1 res 58.9% |
| 4. Classical vs deep vs hybrid | Classical: 14.7% | Deep learning: 94.6% | Hybrid: 94.7% |
| 5. Ablation | 11 configurations in metrics.json['ablation'] |
| 6. Optimisation | INT8 ONNX 34 MB vs FP32 112 MB, top-1 94.7% vs 94.6%; FP16 + pruning also reported |
| 7. Unseen test images | all metrics on 8041 test images never used for training/tuning |
| 8. Failure cases | 8 explained examples; causes: {'same_make': 219, 'low_resolution': 28, 'cross_make_lookalike': 98, 'truncation/occlusion': 39, 'year_confusion': 37, 'localisation': 2, 'low_light': 6} |
