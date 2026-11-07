"""Rebuild artifacts/config.json and artifacts/class_meta.json exactly as notebook cell 13.1 would have.

The Colab run stopped before section 13, so these two files were never exported. Values come from:
  - CFG in notebook cell 1 (crop_pad, det_conf, yolo_imgsz, dark/bright thresholds)
  - classifier_best.pth (model_name, num_classes, img_size, mean, std)
  - metrics.json -> calibration_params (T, alpha, T2)
  - tau / tau_make reported by notebook cell 10.2 (passed on the command line)
  - cars_meta.mat from the same dataset mirror the notebook cloned (class order = class id)

Usage:  python scripts/build_config.py --tau 0.14 --tau-make 0.00
"""
import argparse, json, os, sys

import scipy.io as sio
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, 'artifacts')

# --- notebook cell 2.2, verbatim ---
MULTI_WORD_MAKES = ['AM General', 'Aston Martin', 'Land Rover']
def parse_name(n):
    parts = n.split()
    year = parts[-1] if parts[-1].isdigit() else ''
    body = ' '.join(parts[:-1] if year else parts)
    for mk in MULTI_WORD_MAKES:
        if body.startswith(mk + ' '):
            return mk, body[len(mk) + 1:], year
    mk, _, rest = body.partition(' ')
    return mk, rest, year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tau', type=float, required=True)
    ap.add_argument('--tau-make', type=float, required=True)
    ap.add_argument('--meta-mat', default=os.path.join(ROOT, 'scripts', 'data', 'cars_meta.mat'))
    a = ap.parse_args()

    names = [str(x) for x in sio.loadmat(a.meta_mat, squeeze_me=True)['class_names']]
    class_meta = []
    for i, n in enumerate(names):
        mk, md, yr = parse_name(n)
        class_meta.append(dict(id=i, name=n, make=mk, model=md, year=yr))

    ck = torch.load(os.path.join(ART, 'classifier_best.pth'), map_location='cpu', weights_only=False)
    assert ck['num_classes'] == len(class_meta), (ck['num_classes'], len(class_meta))
    cal = json.load(open(os.path.join(ART, 'metrics.json')))['calibration_params']

    config = dict(model_name=ck['model_name'], num_classes=ck['num_classes'], img_size=ck['img_size'],
                  mean=list(ck['mean']), std=list(ck['std']), crop_pad=0.08,
                  T=cal['T'], alpha=cal['alpha'], T2=cal['T2'], tau=a.tau, tau_make=a.tau_make,
                  det_conf=0.25, yolo_imgsz=640, enhance=True, dark_thr=80, bright_thr=200, tta=True)

    json.dump(config, open(os.path.join(ART, 'config.json'), 'w'), indent=2)
    json.dump(class_meta, open(os.path.join(ART, 'class_meta.json'), 'w'), indent=1)
    print(json.dumps(config, indent=2))
    print(f'{len(class_meta)} classes, e.g.', class_meta[0], class_meta[-1], sep='\n  ')
    print('checkpoint: epoch', ck.get('epoch'), 'val_top1', ck.get('val_top1'))


if __name__ == '__main__':
    sys.exit(main())
