"""Smoke test: load VehicleRecognizer from ./artifacts, run predict() on sample image(s), print the results.

    python scripts/smoke_test.py                       # all images in ./samples
    python scripts/smoke_test.py path/to/car.jpg       # specific image(s)
    BACKEND=onnx python scripts/smoke_test.py          # ONNX INT8 classifier (needs classifier_int8.onnx)
"""
import glob, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, 'artifacts')
sys.path.insert(0, ART)  # inference.py imports classical_cv from the same folder

import torch  # noqa: E402
from inference import VehicleRecognizer  # noqa: E402


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, 'samples', '*.jpg')))
    if not paths:
        sys.exit('No images given and ./samples is empty.')
    gt_file = os.path.join(ROOT, 'samples', 'ground_truth.json')
    gt = json.load(open(gt_file)) if os.path.exists(gt_file) else {}
    backend = os.environ.get('BACKEND', 'torch')

    print(f'torch {torch.__version__} | cuda={torch.cuda.is_available()} | backend={backend}')
    t0 = time.time()
    rec = VehicleRecognizer(ART, backend=backend)
    print(f'Loaded in {time.time() - t0:.1f}s on device={rec.device}')

    use_cam = backend == 'torch'
    rec.predict(paths[0], return_stages=False)  # warm-up
    correct = 0
    for p in paths:
        out = rec.predict(p, max_vehicles=5, return_stages=True, gradcam=use_cam)
        name = os.path.basename(p)
        print(f'\n=== {name}  (ground truth: {gt.get(name, "?")})')
        for v in out['vehicles']:
            print(f"  [{v['status']}] {v['make']} | {v['model']} | {v['year']}  conf={v['confidence']:.3f}  "
                  f"det={v['det_source']}:{v['det_conf']:.2f}  box={[round(b) for b in v['box']]}")
            print(f"    message: {v['message']}")
            print('    top5:', ', '.join(f"{t['label']} {t['confidence']:.2f}" for t in v['top5']))
        correct += gt.get(name) == out['vehicles'][0]['label']
        print('  annotated:', out['annotated_image'].shape, out['annotated_image'].dtype,
              '| gradcam:', None if out.get('gradcam') is None else out['gradcam'].shape)
        print('  stages:', list(out['stages']))
        print('  classical_info:', {k: (round(float(v), 3) if isinstance(v, (int, float)) or hasattr(v, 'item') else v)
                                    for k, v in out['classical_info'].items()})
        print('  timings_ms:', {k: round(v, 1) for k, v in out['timings_ms'].items()})
    if gt:
        print(f'\nTop-1 on samples with ground truth: {correct}/{sum(os.path.basename(p) in gt for p in paths)}')


if __name__ == '__main__':
    main()
