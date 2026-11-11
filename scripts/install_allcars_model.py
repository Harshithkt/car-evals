"""Install a finished all-cars training run into artifacts_modern/ (the folder the backend serves as the "allcars" model).

    .venv/bin/modal run modal_train/train_modern.py::fetch --dest /tmp/allcars_run
    python scripts/install_allcars_model.py /tmp/allcars_run --gpu H100 --cost-usd 5.80

What it does
  * copies the classifier (PyTorch + ONNX INT8), config.json, class_meta.json, metrics.json, sample and failure crops,
    credits.json and the split index into artifacts_modern/ (the old folder is kept as artifacts_modern.bak-<time>/)
  * copies the shared YOLOv8 car detector from artifacts/ (inference.py loads it from the model's own folder)
  * adds the GPU benchmark (modal_train/gpu_benchmark.json) to metrics.json
  * --gpu / --cost-usd overwrite the GPU name and total cost in metrics.json with the real values from `modal billing report`
"""
import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path, help='folder written by train_modern.py::fetch (has artifacts/, samples/, failures/)')
    ap.add_argument('--dst', type=Path, default=ROOT / 'artifacts_modern')
    ap.add_argument('--gpu', help='GPU the run used (as named in Modal pricing)')
    ap.add_argument('--cost-usd', type=float, help='total Modal cost of benchmark + prep + smoke tests + training')
    ap.add_argument('--total-minutes', type=float, help='GPU wall time of all training runs together (if the run was resumed)')
    ap.add_argument('--note', help='free-text note stored in metrics.training.note')
    a = ap.parse_args()

    src = a.src.resolve()
    need = [src / 'artifacts' / f for f in ('classifier_int8.onnx', 'config.json', 'class_meta.json', 'metrics.json')]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise SystemExit(f'missing: {missing}')
    metrics = json.loads((src / 'artifacts' / 'metrics.json').read_text())
    if metrics.get('smoke_test'):
        raise SystemExit('this is a smoke-test run, not a real model')

    if a.dst.exists():
        bak = a.dst.with_name(f'{a.dst.name}.bak-{time.strftime("%Y%m%d-%H%M%S")}')
        a.dst.rename(bak)
        print('previous model kept in', bak)
    a.dst.mkdir(parents=True)
    for f in (src / 'artifacts').iterdir():
        shutil.copy2(f, a.dst / f.name)
    for d in ('samples', 'failures'):
        if (src / d).is_dir():
            shutil.copytree(src / d, a.dst / d)
    for f in ('credits.json', 'split_index.json.gz'):
        if (src / f).exists():
            shutil.copy2(src / f, a.dst / f)
    shutil.copy2(ROOT / 'artifacts' / 'yolo_car_best.pt', a.dst / 'yolo_car_best.pt')

    bench = ROOT / 'modal_train' / 'gpu_benchmark.json'
    if bench.exists():
        metrics['gpu_benchmark'] = json.loads(bench.read_text())['rows']
    tr = metrics.setdefault('training', {})
    if a.gpu:
        tr['gpu'] = a.gpu
    if a.cost_usd is not None:
        tr['cost_usd_total_estimate'] = round(a.cost_usd, 2)
        tr['cost_source'] = 'modal billing report (benchmark + data prep + smoke tests + training)'
    if a.total_minutes is not None:
        tr['total_minutes'] = round(a.total_minutes, 1)
        tr['train_minutes'] = round(sum(h.get('train_sec', 0) for h in tr.get('history', [])) / 60, 1)
    if a.note:
        tr['note'] = a.note
    (a.dst / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    t = metrics.get('test', {})
    print(f"installed {len(json.loads((a.dst / 'class_meta.json').read_text()))} classes into {a.dst}: "
          f"top-1 {t.get('top1', 0):.4f}, top-5 {t.get('top5', 0):.4f}, make {t.get('make_acc', 0):.4f}")


if __name__ == '__main__':
    main()
