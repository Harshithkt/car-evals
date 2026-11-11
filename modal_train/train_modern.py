"""Train the "all cars" classifier on Modal: ~1,300 car generations from 1948 to 2026.

Run from the project folder, in this order:

    .venv/bin/modal run modal_train/train_modern.py::bench            # 1. GPU speed and price benchmark (about $0.40)
    .venv/bin/modal run modal_train/train_modern.py::prep             # 2. download and crop the dataset into the volume (CPU only)
    .venv/bin/modal run modal_train/train_modern.py::train --smoke    # 3. end-to-end smoke test on a few makes (~$0.10)
    P5_GPU=H100 .venv/bin/modal deploy modal_train/train_modern.py    # 4. train, calibrate, export ONNX INT8 (GPU) ...
    .venv/bin/python -c "import modal; print(modal.Function.from_name('p5-train-modern', 'train_remote').spawn(epochs=20, gpu_name='H100').object_id)"
    .venv/bin/modal run modal_train/train_modern.py::fetch --dest /tmp/allcars_run   # 5. download the results
    .venv/bin/python scripts/install_allcars_model.py /tmp/allcars_run               # 6. install into artifacts_modern/

Step 4 deploys the app and *spawns* the training call, so it keeps running even if the local terminal closes (a call made
with `modal run` is cancelled when the local client exits, even with --detach). The run checkpoints every epoch and
resumes from the last one if it is started again with the same settings. Stop the app afterwards: modal app stop p5-train-modern.

Dataset: timdwms/us-car-models-1948-2026 on Hugging Face. These are Wikimedia Commons photos, and each one
keeps its own Creative Commons licence (see the dataset card). Only photos that passed the dataset's quality
filter are used. Each photo is cropped to its car box (plus padding) and stored as a small JPEG, so training
reads about 7 GB instead of 42 GB.

Spending is capped in three ways: a hard function timeout, a time-based budget check after every epoch, and
`max_containers=1` on the GPU function.
"""
import json
import os
import pathlib
import time

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = '/data'
HF_REPO = 'timdwms/us-car-models-1948-2026'
HF_REV = '063a8a846d4797c520447bf29ddf51318a4e792c'   # pinned dataset commit (2026-10-29)
MODEL_NAME = 'convnext_tiny.fb_in22k_ft_in1k'          # same backbone as the Stanford Cars model
IMG = 288
CROP_PAD = 0.08            # padding the app uses around a detected car
STORE_PAD = 0.15           # extra context kept in the stored crops (room for box-jitter augmentation)
STORE_SHORT, STORE_LONG = 320, 720
MIN_PER_CLASS = 30         # generations with fewer clean photos are left out

TRAIN_GPU = os.environ.get('P5_GPU', 'A10')
TRAIN_CPU, TRAIN_MEM_GIB = 8, 24

# Modal list prices in USD per second (modal.com/pricing, checked 2026-11-07). No region pinning, so no multiplier.
GPU_PRICE = {'T4': 0.000164, 'L4': 0.000222, 'A10': 0.000306, 'L40S': 0.000542,
             'A100-40GB': 0.000583, 'A100-80GB': 0.000694, 'H100': 0.001097}
CPU_CORE_S, MEM_GIB_S = 0.0000131, 0.00000222


def rate_per_s(gpu, cpu=TRAIN_CPU, mem_gib=TRAIN_MEM_GIB):
    return GPU_PRICE[gpu] + cpu * CPU_CORE_S + mem_gib * MEM_GIB_S


app = modal.App('p5-train-modern')
vol = modal.Volume.from_name('p5-modern', create_if_missing=True)

cpu_image = modal.Image.debian_slim(python_version='3.11').pip_install('huggingface_hub[hf_xet]', 'Pillow', 'numpy')
gpu_image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('libgl1', 'libglib2.0-0')
    .pip_install('torch==2.8.0', 'torchvision==0.23.0')
    .pip_install('timm', 'onnx', 'onnxruntime', 'onnxscript', 'opencv-python-headless', 'scikit-image',
                 'numpy', 'Pillow', 'huggingface_hub[hf_xet]')
    .add_local_file(ROOT / 'artifacts' / 'classical_cv.py', '/opt/p5/classical_cv.py')
)


# --------------------------------------------------------------------------------------------------------------
# 1. GPU benchmark: same model, input size and precision as the real run, synthetic data (pure GPU speed)
# --------------------------------------------------------------------------------------------------------------
@app.function(image=gpu_image, gpu='T4', cpu=2.0, memory=8192, timeout=900)
def bench_gpu(name: str, batch: int = 64, steps: int = 30) -> dict:
    import torch
    import torch.nn.functional as F
    import timm
    torch.backends.cudnn.benchmark = True
    m = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1300, drop_path_rate=0.1).cuda()
    m = m.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    dt = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=not bf16)
    x = torch.randn(batch, 3, IMG, IMG, device='cuda').contiguous(memory_format=torch.channels_last)
    y = torch.randint(0, 1300, (batch,), device='cuda')

    def step():
        with torch.autocast('cuda', dtype=dt):
            loss = F.cross_entropy(m(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(8):
        step()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    ips = steps * batch / (time.perf_counter() - t)
    return dict(gpu=name, device=torch.cuda.get_device_name(), dtype=str(dt).split('.')[-1], img_per_s=round(ips, 1),
                peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))


@app.local_entrypoint()
def bench(gpus: str = 'T4,L4,A10,L40S,A100-40GB,H100'):
    calls = {g: bench_gpu.with_options(gpu=g).spawn(g) for g in gpus.split(',')}
    rows = []
    for g, c in calls.items():
        try:
            r = c.get()
        except Exception as e:  # noqa: BLE001
            print(f'{g}: failed ({e})')
            continue
        rate = rate_per_s(g)
        r['usd_per_hour_with_8cpu_24gb'] = round(rate * 3600, 3)
        r['usd_per_1M_train_images'] = round(1e6 / r['img_per_s'] * rate, 3)
        rows.append(r)
    rows.sort(key=lambda r: r['usd_per_1M_train_images'])
    print(f"{'GPU':<10}{'device':<34}{'img/s':>8}{'$/h':>8}{'$ per 1M imgs':>15}")
    for r in rows:
        print(f"{r['gpu']:<10}{r['device'][:33]:<34}{r['img_per_s']:>8}{r['usd_per_hour_with_8cpu_24gb']:>8}"
              f"{r['usd_per_1M_train_images']:>15}")
    out = ROOT / 'modal_train' / 'gpu_benchmark.json'
    out.write_text(json.dumps(dict(model=MODEL_NAME, img_size=IMG, batch=64, prices_checked='2026-11-07', rows=rows), indent=2))
    print('saved', out)


def crop_to_jpeg(data: bytes, car_box):
    """Crop a photo to its car box plus STORE_PAD, shrink it, return (jpeg_bytes, box_inside_crop, w, h)."""
    import io
    import math
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(data))
        im = im.convert('RGB')
    except Exception:  # noqa: BLE001
        return None
    W, H = im.size
    x1, y1, x2, y2 = car_box
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 40 or bh < 30:
        return None
    box = (int(max(0, x1 - STORE_PAD * bw)), int(max(0, y1 - STORE_PAD * bh)),
           int(math.ceil(min(W, x2 + STORE_PAD * bw))), int(math.ceil(min(H, y2 + STORE_PAD * bh))))
    cw, ch = box[2] - box[0], box[3] - box[1]
    s = min(1.0, STORE_SHORT / min(cw, ch), STORE_LONG / max(cw, ch))
    tw, th = max(1, round(cw * s)), max(1, round(ch * s))
    crop = im.resize((tw, th), Image.BICUBIC, box=box, reducing_gap=3.0)
    sx, sy = tw / cw, th / ch
    rel = [round((x1 - box[0]) * sx, 1), round((y1 - box[1]) * sy, 1), round((x2 - box[0]) * sx, 1), round((y2 - box[1]) * sy, 1)]
    buf = io.BytesIO()
    crop.save(buf, 'JPEG', quality=90)
    return buf.getvalue(), rel, tw, th


# --------------------------------------------------------------------------------------------------------------
# 2. Data prep: one CPU container per make. Download the make's zip, crop every clean photo to its car box.
#    Output per make: shards/<make>.bin (JPEG bytes back to back) + shards/<make>.json (offsets and labels).
# --------------------------------------------------------------------------------------------------------------
@app.function(image=cpu_image, cpu=2.0, memory=8192, timeout=3600, volumes={DATA: vol}, retries=1)
def prep_make(zip_name: str, rows: list) -> dict:
    import zipfile
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    make = zip_name[:-4]
    os.makedirs(f'{DATA}/shards', exist_ok=True)
    out_bin, out_idx = f'{DATA}/shards/{make}.bin', f'{DATA}/shards/{make}.json'
    if os.path.exists(out_idx):
        return json.load(open(out_idx))['summary']
    path = hf_hub_download(HF_REPO, f'images/{zip_name}', repo_type='dataset', revision=HF_REV, local_dir='/tmp/dl')
    t_dl = time.time() - t0
    zf = zipfile.ZipFile(path)
    names = set(zf.namelist())

    def work(item):
        r, data = item
        res = crop_to_jpeg(data, [float(v) for v in r['car_box'].split()])
        return None if res is None else (r, *res)

    recs, off, skipped, missing = [], 0, 0, 0
    with open(out_bin + '.tmp', 'wb') as fb, ThreadPoolExecutor(4) as pool:
        for i in range(0, len(rows), 256):
            chunk = []
            for r in rows[i:i + 256]:
                if r['file_name'] in names:
                    chunk.append((r, zf.read(r['file_name'])))
                else:
                    missing += 1
            for res in pool.map(work, chunk):
                if res is None:
                    skipped += 1
                    continue
                r, jpg, rel, tw, th = res
                fb.write(jpg)
                recs.append(dict(f=r['file_name'], o=off, n=len(jpg), b=rel, w=tw, h=th, make=r['make'], model=r['model'],
                                 gen=r['generation'], years=r['years'], author=r['author'], date=r['date_taken'],
                                 lic=r['license'], url=r['commons_url'], src='commons-hf'))
                off += len(jpg)
    os.replace(out_bin + '.tmp', out_bin)
    summary = dict(make=make, n=len(recs), skipped=skipped, missing=missing, mb=round(off / 1e6, 1),
                   sec_download=round(t_dl), sec_total=round(time.time() - t0))
    json.dump(dict(summary=summary, records=recs), open(out_idx, 'w'))
    vol.commit()
    return summary


@app.function(image=cpu_image, cpu=1.0, memory=4096, timeout=2 * 3600, volumes={DATA: vol})
def prep_all() -> list:
    import collections
    import csv
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(HF_REPO, 'metadata.csv', repo_type='dataset', revision=HF_REV, local_dir='/tmp/meta')
    keep = ('file_name', 'make', 'model', 'generation', 'years', 'car_box', 'author', 'date_taken', 'license', 'commons_url')
    by_zip = collections.defaultdict(list)
    for r in csv.DictReader(open(p, newline='', encoding='utf-8')):
        if r['filter'] == 'ok' and r['zip'] != '_Multiple cars.zip' and r['car_box'].strip():
            by_zip[r['zip']].append({k: r[k] for k in keep})
    jobs = sorted(by_zip.items(), key=lambda kv: -len(kv[1]))   # biggest makes first
    out = []
    for (z, _), res in zip(jobs, prep_make.starmap(jobs, return_exceptions=True)):
        out.append(res if isinstance(res, dict) else dict(make=z, error=repr(res)))
        print(out[-1], flush=True)
    return out


@app.local_entrypoint()
def prep():
    t = time.time()
    res = prep_all.remote()
    ok = [r for r in res if 'error' not in r]
    print(f"\n{len(ok)}/{len(res)} makes done, {sum(r['n'] for r in ok):,} crops, {sum(r['mb'] for r in ok) / 1e3:.2f} GB, "
          f"{(time.time() - t) / 60:.1f} min")
    for r in res:
        if 'error' in r:
            print('FAILED', r)


# Makes that Stanford Cars has and the Commons dataset does not (mostly exotic and luxury brands). Only Stanford
# *train* images are used, so the Stanford test set stays unseen and the original benchmark numbers stay valid.
STANFORD_EXTRA_MAKES = ['AM General', 'Aston Martin', 'Bentley', 'Bugatti', 'Daewoo', 'Eagle', 'Ferrari', 'Fisker', 'Geo',
                        'Lamborghini', 'Maybach', 'McLaren', 'Rolls-Royce', 'Spyker']
stanford_image = cpu_image.apt_install('git').pip_install('scipy')


@app.function(image=stanford_image, cpu=2.0, memory=4096, timeout=1800, volumes={DATA: vol})
def prep_stanford() -> dict:
    import subprocess
    from scipy.io import loadmat
    t0 = time.time()
    out_bin, out_idx = f'{DATA}/shards/_stanford_extra.bin', f'{DATA}/shards/_stanford_extra.json'
    os.makedirs(f'{DATA}/shards', exist_ok=True)
    src = '/tmp/sc'
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/jhpohovey/StanfordCars-Dataset.git', src], check=True)
    names = [str(x[0]) for x in loadmat(f'{src}/stanford_cars/devkit/cars_meta.mat')['class_names'][0]]
    ann = loadmat(f'{src}/stanford_cars/devkit/cars_train_annos.mat')['annotations'][0]
    recs, off = [], 0
    with open(out_bin, 'wb') as fb:
        for a in ann:
            name = names[int(a['class'][0][0]) - 1]
            make = next((m for m in STANFORD_EXTRA_MAKES if name.startswith(m + ' ')), None)
            if make is None:
                continue
            rest, year = name[len(make) + 1:].rsplit(' ', 1)
            fname = str(a['fname'][0])
            box = [float(a[k][0][0]) for k in ('bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2')]
            res = crop_to_jpeg(open(f'{src}/stanford_cars/cars_train/{fname}', 'rb').read(), box)
            if res is None:
                continue
            jpg, rel, tw, th = res
            fb.write(jpg)
            recs.append(dict(f=f'stanford_train/{fname}', o=off, n=len(jpg), b=rel, w=tw, h=th, make=make, model=rest, gen='',
                             years=year, author='Stanford Cars (Krause et al., 2013)', date='', lic='stanford-cars-research',
                             url='https://ai.stanford.edu/~jkrause/cars/car_dataset.html', src='stanford-train'))
            off += len(jpg)
    summary = dict(make='_stanford_extra', n=len(recs), classes=len({(r['make'], r['model'], r['years']) for r in recs}),
                   mb=round(off / 1e6, 1), sec_total=round(time.time() - t0))
    json.dump(dict(summary=summary, records=recs), open(out_idx, 'w'))
    vol.commit()
    return summary


@app.local_entrypoint()
def prep_extra():
    print(prep_stanford.remote())


# --------------------------------------------------------------------------------------------------------------
# 3. Classes and splits
# --------------------------------------------------------------------------------------------------------------
MODEL_FIX = {
    'Ilx': 'ILX', 'Cr-V': 'CR-V', 'Cr-Z': 'CR-Z', 'Hr-V': 'HR-V', 'Delsol': 'del Sol', 'E-Tron': 'e-tron', 'E-Tron GT': 'e-tron GT',
    'Q4 E-Tron': 'Q4 e-tron', 'I3': 'i3', 'I8': 'i8', 'Ix': 'iX', 'Parkavenue': 'Park Avenue', 'Lacrosse': 'LaCrosse',
    'Lesabre': 'LeSabre', 'Ats': 'ATS', 'Elr': 'ELR', 'Blazer Ev': 'Blazer EV', 'Equinox Ev': 'Equinox EV',
    'Silverado Ev': 'Silverado EV', 'Montecarlo': 'Monte Carlo', 'Town&Country': 'Town & Country', 'RAM_1500': 'Ram 1500',
    'Ram_Hd': 'Ram HD', 'Ecosport': 'EcoSport', 'Hummer Ev': 'Hummer EV', 'SIERRA_1500': 'Sierra 1500',
    'SUBURBAN_1500': 'Suburban 1500', 'Sierra_Hd': 'Sierra HD', 'YUKON_1500': 'Yukon 1500', 'Santafe': 'Santa Fe', 'Xe': 'XE',
    'Grand_Cherokee': 'Grand Cherokee', 'Discovery_Sport': 'Discovery Sport', 'Rangerover': 'Range Rover',
    'Rangerover_Evoque': 'Range Rover Evoque', 'Rangerover_Sport': 'Range Rover Sport', 'Rangerover_Velar': 'Range Rover Velar',
    'Ux': 'UX', 'Towncar': 'Town Car', 'CX3': 'CX-3', 'CX30': 'CX-30', 'CX5': 'CX-5', 'CX50': 'CX-50', 'CX7': 'CX-7',
    'CX9': 'CX-9', 'CX90': 'CX-90', 'MX5_MIATA': 'MX-5 Miata', 'Amg GT': 'AMG GT', 'Cla-Class': 'CLA-Class',
    'Cls-Class': 'CLS-Class', 'Eqe': 'EQE', 'Eqe Suv': 'EQE SUV', 'Eqs': 'EQS', 'Eqs Suv': 'EQS SUV', 'Gl-Class': 'GL-Class',
    'Gla-Class': 'GLA-Class', 'Glb-Class': 'GLB-Class', 'Glc-Class': 'GLC-Class', 'Gle-Class': 'GLE-Class',
    'Glk-Class': 'GLK-Class', 'Gls-Class': 'GLS-Class', 'Slk-Class': 'SLK-Class', 'Grandmarquis': 'Grand Marquis',
    'I-Miev': 'i-MiEV', 'GTR': 'GT-R', 'Grandam': 'Grand Am', 'Grandprix': 'Grand Prix', 'Gto': 'GTO', 'Transam': 'Trans Am',
    'Polestar 2': '2', 'Polestar 3': '3', 'Iq': 'iQ', 'TC': 'tC', 'XB': 'xB', 'XD': 'xD', 'FRS': 'FR-S',
    'Grand_Vitara': 'Grand Vitara', '4RUNNER': '4Runner', 'BZ4X': 'bZ4X', 'C-Hr': 'C-HR', 'Fjcruiser': 'FJ Cruiser',
    'Landcruiser': 'Land Cruiser', 'Transit_Connect': 'Transit Connect',
}


def display_model(model, gen=''):
    import re
    m = MODEL_FIX.get(model, model.replace('_', ' '))
    g = (gen or '').strip()
    if not g or re.fullmatch(r'\d{4}', g):
        return m
    if g.lower().startswith(m.lower()):          # "HHR" + "HHR" -> "HHR", "Golf" + "Golf VII" -> "Golf VII"
        return g if len(g) > len(m) else m
    return f'{m} {g}'


def display_years(y):
    a, _, b = (y or '').partition('-')
    return a if (not b or a == b) else f'{a}-{b}'


def start_year(y):
    import re
    m = re.match(r'(\d{4})', y or '')
    return int(m.group(1)) if m else 0


def build_dataset(recs, seed=42):
    """Classes = car generations with enough clean photos. Split 80/10/10 inside each class, keeping all photos
    by the same photographer on the same day in one split (they are usually the same car), so near-duplicate
    shots cannot leak from train into test."""
    import collections
    import numpy as np
    key = [(r['make'], r['model'], r['gen'], r['years']) for r in recs]
    src = {k: r['src'] for k, r in zip(key, recs)}
    cnt = collections.Counter(key)
    keep = [k for k, c in cnt.items() if c >= (20 if src[k] == 'stanford-train' else MIN_PER_CLASS)]
    keep.sort(key=lambda k: (k[0].lower(), display_model(k[1]).lower(), start_year(k[3]), k[2]))
    cid = {k: i for i, k in enumerate(keep)}
    classes = []
    for i, k in enumerate(keep):
        make, model, gen, years = k
        md, yd = display_model(model, gen), display_years(years)
        classes.append(dict(id=i, name=f'{make} {md} {yd}', make=make, model=md, year=yd, base_model=display_model(model),
                            generation=gen, start_year=start_year(years), source=src[k], n_images=cnt[k]))
    y = np.array([cid.get(k, -1) for k in key])
    split = np.full(len(recs), -1)
    rng = np.random.default_rng(seed)
    by_class = collections.defaultdict(list)
    for i, c in enumerate(y):
        if c >= 0:
            by_class[int(c)].append(i)
    fallback = 0
    for c in range(len(classes)):
        idx = by_class[c]
        groups = collections.defaultdict(list)
        for i in idx:
            r = recs[i]
            groups[(r['author'], r['date'][:10]) if r['date'] else (r['author'], r['f'])].append(i)
        g = list(groups.values())
        rng.shuffle(g)
        n = len(idx)
        n_te = n_va = max(2, round(0.1 * n))
        te, va, tr = [], [], []
        for grp in g:
            if len(te) < n_te and len(te) + len(grp) <= max(3, 1.6 * n_te):
                te += grp
            elif len(va) < n_va and len(va) + len(grp) <= max(3, 1.6 * n_va):
                va += grp
            else:
                tr += grp
        if len(te) < 2 or len(va) < 2 or len(tr) < 0.6 * n:      # one photographer dominates: plain random split
            fallback += 1
            p = rng.permutation(idx).tolist()
            te, va, tr = p[:n_te], p[n_te:n_te + n_va], p[n_te + n_va:]
        split[te], split[va], split[tr] = 2, 1, 0
    return classes, y, split, fallback


# --------------------------------------------------------------------------------------------------------------
# 4. Training (GPU). Light work (JPEG decode, crop, flip) runs in CPU threads; the rest of the augmentation runs
#    on the GPU in batches, so 8 CPU cores are enough to keep the GPU busy.
# --------------------------------------------------------------------------------------------------------------
SPENT_BEFORE_TRAIN = 2.1   # measured with `modal billing report`: benchmark, prep, smoke tests and the first 4 epochs (2.02)
BUDGET_USD = 9.0           # stays under the $10 cap
TRAIN_TIMEOUT = int(min(4 * 3600, (BUDGET_USD - SPENT_BEFORE_TRAIN) / rate_per_s(TRAIN_GPU)))


@app.function(image=gpu_image, gpu=TRAIN_GPU, cpu=float(TRAIN_CPU), memory=TRAIN_MEM_GIB * 1024, timeout=TRAIN_TIMEOUT,
              volumes={DATA: vol}, max_containers=1)
def train_remote(epochs: int = 12, batch: int = 128, lr: float = 4e-4, budget_usd: float = BUDGET_USD,
                 spent_usd: float = SPENT_BEFORE_TRAIN, smoke: bool = False, resume: bool = True, compile_model: bool = True,
                 gpu_name: str = '') -> dict:
    import collections
    import copy
    import glob
    import gzip
    import io
    import math
    import random
    import sys
    from concurrent.futures import ThreadPoolExecutor

    import cv2
    import numpy as np
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    sys.path.insert(0, '/opt/p5')
    from classical_cv import adaptive_enhance

    T0 = time.time()
    # the container re-imports this file without P5_GPU, so the caller passes the GPU name for the price
    gpu = gpu_name or TRAIN_GPU
    rate = rate_per_s(gpu)

    def spent():
        return spent_usd + (time.time() - T0) * rate

    def log(*a):
        print(f'[{(time.time() - T0) / 60:6.1f} min | ${spent():.2f}]', *a, flush=True)

    RUN = f'{DATA}/smoke' if smoke else f'{DATA}/run'
    OUT = f'{RUN}/out'
    os.makedirs(f'{OUT}/artifacts', exist_ok=True)
    os.makedirs(f'{OUT}/samples', exist_ok=True)
    os.makedirs(f'{OUT}/failures', exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    amp_dt = torch.bfloat16 if bf16 else torch.float16

    # ---------------- data ----------------
    shards, recs = [], []
    for j in sorted(glob.glob(f'{DATA}/shards/*.json')):
        d = json.load(open(j))
        if smoke and d['summary']['make'] not in ('Tesla', 'Rivian', 'Lucid', 'Polestar', 'Smart', 'Ram', '_stanford_extra'):
            continue
        sid = len(shards)
        shards.append(np.fromfile(j[:-5] + '.bin', dtype=np.uint8))
        for r in d['records']:
            r['s'] = sid
            recs.append(r)
    classes, y, split, fallback = build_dataset(recs)
    C = len(classes)
    tr, va, te = (np.where(split == s)[0] for s in (0, 1, 2))
    if smoke:
        va, te = va[:400], te[:400]
    log(f'{len(recs):,} crops in RAM ({sum(s.nbytes for s in shards) / 1e9:.2f} GB) | {C} classes | '
        f'train {len(tr):,} val {len(va):,} test {len(te):,} | random-split fallback in {fallback} classes')
    n_per = np.bincount(y[y >= 0], minlength=C)
    for c in classes:
        i = c['id']
        c['n_train'] = int(((y == i) & (split == 0)).sum())
        c['n_test'] = int(((y == i) & (split == 2)).sum())
    del n_per

    def decode(i):
        r = recs[i]
        im = Image.open(io.BytesIO(shards[r['s']][r['o']:r['o'] + r['n']].tobytes())).convert('RGB')
        return im, r['b']

    def load_train(i, seed):
        rng = np.random.default_rng(seed)
        im, (x1, y1, x2, y2) = decode(i)
        W, H = im.size
        bw, bh = x2 - x1, y2 - y1
        p = rng.uniform(0.0, 0.15)                              # padding around the car (the app uses 0.08)
        j = rng.uniform(-0.06, 0.06, 4) * np.array([bw, bh, bw, bh])   # detector box wobble
        bx1, by1 = max(0.0, x1 - p * bw + j[0]), max(0.0, y1 - p * bh + j[1])
        bx2, by2 = min(float(W), x2 + p * bw + j[2]), min(float(H), y2 + p * bh + j[3])
        area, ratio = rng.uniform(0.6, 1.0), math.exp(rng.uniform(math.log(3 / 4), math.log(4 / 3)))
        cw, ch = (bx2 - bx1) * min(1.0, math.sqrt(area * ratio)), (by2 - by1) * min(1.0, math.sqrt(area / ratio))
        cx = bx1 + rng.uniform(0.0, max(0.0, (bx2 - bx1) - cw))   # max(): float rounding can make the range -1e-13
        cy = by1 + rng.uniform(0.0, max(0.0, (by2 - by1) - ch))
        a = np.asarray(im.resize((IMG, IMG), Image.BILINEAR, box=(cx, cy, cx + cw, cy + ch)))
        return a[:, ::-1] if rng.random() < 0.5 else a

    def load_eval(i, enhance):
        """Exactly what the app does: crop with CROP_PAD, illumination fix on dark/bright crops, squash to IMG x IMG."""
        im, (x1, y1, x2, y2) = decode(i)
        W, H = im.size
        bw, bh = x2 - x1, y2 - y1
        box = (int(max(0, x1 - CROP_PAD * bw)), int(max(0, y1 - CROP_PAD * bh)), int(min(W, x2 + CROP_PAD * bw)), int(min(H, y2 + CROP_PAD * bh)))
        if not enhance:
            return np.asarray(im.resize((IMG, IMG), Image.BILINEAR, box=box))
        bgr = cv2.cvtColor(np.asarray(im.crop(box)), cv2.COLOR_RGB2BGR)
        bgr = adaptive_enhance(bgr, 80, 200)[0]
        return np.asarray(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((IMG, IMG), Image.BILINEAR))

    pool = ThreadPoolExecutor(TRAIN_CPU * 2)

    def batches(fn, idx, bs, shuffle=False, drop_last=False, epoch=0, ahead=6):
        order = np.random.default_rng(1000 + epoch).permutation(idx) if shuffle else np.asarray(idx)
        chunks = [order[k:k + bs] for k in range(0, len(order), bs)]
        if drop_last and chunks and len(chunks[-1]) < bs:
            chunks.pop()
        it = iter(chunks)
        pending = collections.deque()

        def submit(b):
            return b, [pool.submit(fn, int(i), (epoch, int(i))) for i in b]

        for b in (next(it, None) for _ in range(ahead)):
            if b is not None:
                pending.append(submit(b))
        while pending:
            b, futs = pending.popleft()
            arr = np.stack([f.result() for f in futs])
            nb = next(it, None)
            if nb is not None:
                pending.append(submit(nb))
            yield torch.from_numpy(arr).pin_memory(), torch.from_numpy(y[b])

    MEAN = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    def to_input(xb):
        x = xb.cuda(non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)

    def gpu_augment(xb):
        """Same recipe as the Stanford Cars model: colour jitter, grayscale, small rotation and shear, low resolution,
        blur, random erasing. Done per sample on the GPU."""
        x = xb.cuda(non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        B, dev = x.shape[0], x.device

        def pick(p):
            return torch.rand(B, device=dev) < p

        k = pick(0.3).nonzero().squeeze(1)
        if len(k):
            ang = (torch.rand(len(k), device=dev) * 20 - 10) * math.pi / 180
            sh = torch.rand(len(k), device=dev) * 0.2 - 0.1
            sc = 1 + torch.rand(len(k), device=dev) * 0.1
            cos, sin, z = torch.cos(ang) / sc, torch.sin(ang) / sc, torch.zeros(len(k), device=dev)
            theta = torch.stack([torch.stack([cos, -sin + sh, z], 1), torch.stack([sin, cos, z], 1)], 1)
            grid = F.affine_grid(theta, (len(k), 3, IMG, IMG), align_corners=False)
            x[k] = F.grid_sample(x[k], grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        on = pick(0.8).float().view(B, 1, 1, 1)

        def fac(a):
            return 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * a * on

        x = x * fac(0.4)
        mu = x.mean((1, 2, 3), keepdim=True)
        x = (x - mu) * fac(0.4) + mu
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x = (x - gray) * fac(0.3) + gray
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x = torch.where(pick(0.05).view(B, 1, 1, 1), gray.expand_as(x), x).clamp_(0, 1)
        k = pick(0.2).nonzero().squeeze(1)
        if len(k):
            small = F.interpolate(x[k], scale_factor=random.uniform(0.2, 0.6), mode='bilinear', antialias=True, align_corners=False)
            x[k] = F.interpolate(small, size=(IMG, IMG), mode='bilinear', align_corners=False)
        k = pick(0.15).nonzero().squeeze(1)
        if len(k):
            sigma = random.uniform(0.3, 2.0)
            r = int(math.ceil(2 * sigma))
            t = torch.arange(-r, r + 1, device=dev).float()
            w = torch.exp(-t ** 2 / (2 * sigma ** 2))
            w = w / w.sum()
            xk = F.conv2d(F.pad(x[k], (r, r, 0, 0), mode='reflect'), w.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
            x[k] = F.conv2d(F.pad(xk, (0, 0, r, r), mode='reflect'), w.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)
        x = (x - MEAN) / STD
        m = pick(0.35)
        if m.any():
            area = (torch.rand(B, device=dev) * 0.23 + 0.02) * IMG * IMG
            ar = torch.exp(torch.rand(B, device=dev) * (math.log(3.3) - math.log(0.3)) + math.log(0.3))
            h, w = torch.sqrt(area * ar).clamp(1, IMG - 1), torch.sqrt(area / ar).clamp(1, IMG - 1)
            top, left = torch.rand(B, device=dev) * (IMG - h), torch.rand(B, device=dev) * (IMG - w)
            ii, jj = torch.arange(IMG, device=dev).view(1, IMG, 1), torch.arange(IMG, device=dev).view(1, 1, IMG)
            mask = ((ii >= top.view(B, 1, 1)) & (ii < (top + h).view(B, 1, 1)) & (jj >= left.view(B, 1, 1))
                    & (jj < (left + w).view(B, 1, 1)) & m.view(B, 1, 1))
            x = torch.where(mask.unsqueeze(1), torch.randn_like(x), x)
        return x.contiguous(memory_format=torch.channels_last)

    @torch.no_grad()
    def logits(net, idx, enhance=True, tta=True, bs=256):
        net.eval()
        out = []
        for xb, _ in batches(lambda i, s: load_eval(i, enhance), idx, bs):
            x = to_input(xb)
            with torch.autocast('cuda', dtype=amp_dt):
                z = net(x).float()
                if tta:
                    z = (z + net(torch.flip(x, [3])).float()) / 2
            out.append(z.cpu())
        return torch.cat(out).numpy()

    # ---------------- model ----------------
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=C, drop_path_rate=0.1).cuda()
    model = model.to(memory_format=torch.channels_last)
    head = set(id(p) for p in model.get_classifier().parameters())
    groups = collections.defaultdict(list)
    for p in model.parameters():
        groups[(id(p) in head, p.ndim <= 1)].append(p)
    opt = torch.optim.AdamW([dict(params=v, lr=lr * (5 if is_head else 1), weight_decay=0.0 if no_wd else 0.05)
                             for (is_head, no_wd), v in groups.items()], lr=lr, fused=True)
    for g in opt.param_groups:
        g['base_lr'] = g['lr']
    scaler = torch.amp.GradScaler('cuda', enabled=not bf16)
    ema = timm.utils.ModelEmaV3(model, decay=0.9995)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    spe = len(tr) // batch
    max_steps = 40 if smoke else None
    if smoke:
        epochs = 1
    total = epochs * (max_steps or spe)
    warm = min(spe, 800)

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        q = (step - warm) / max(1, total - warm)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, q)))

    start_epoch, step, best, history = 0, 0, dict(acc=-1.0, epoch=-1, which=''), []
    ck_path = f'{RUN}/last.pt'
    if resume and not smoke and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location='cuda', weights_only=False)
        if ck['C'] == C and ck['epochs'] == epochs:
            model.load_state_dict(ck['model'])
            ema.module.load_state_dict(ck['ema'])
            opt.load_state_dict(ck['opt'])
            scaler.load_state_dict(ck['scaler'])
            start_epoch, step, best, history = ck['epoch'] + 1, ck['step'], ck['best'], ck['history']
            log(f'resumed after epoch {start_epoch}')
    run = model
    if compile_model:
        try:
            run = torch.compile(model)
        except Exception as e:  # noqa: BLE001
            log('torch.compile unavailable, eager mode:', e)
    log(f'{MODEL_NAME} | {C} classes | {gpu} {torch.cuda.get_device_name()} | {amp_dt} | batch {batch} | '
        f'{epochs} epochs x {spe} steps | lr {lr}')

    post_s = 300 if smoke else 15 * 60          # time reserved for evaluation + export after training
    stopped_early = ''
    best_sd = None
    for epoch in range(start_epoch, epochs):
        model.train()
        t_ep, n_seen, loss_sum, loss_n = time.time(), 0, torch.zeros((), device='cuda'), 0
        for xb, yb in batches(load_train, tr, batch, shuffle=True, drop_last=True, epoch=epoch):
            for g in opt.param_groups:
                g['lr'] = g['base_lr'] * lr_at(step)
            x, yt = gpu_augment(xb), yb.cuda(non_blocking=True)
            try:
                with torch.autocast('cuda', dtype=amp_dt):
                    loss = crit(run(x), yt)
            except Exception as e:  # noqa: BLE001  (compiler failure -> eager)
                if run is model:
                    raise
                log('compiled step failed, switching to eager:', repr(e)[:300])
                run = model
                with torch.autocast('cuda', dtype=amp_dt):
                    loss = crit(run(x), yt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model, step=step)
            step += 1
            n_seen += len(yb)
            loss_sum += loss.detach()
            loss_n += 1
            if step % 200 == 0:
                log(f'epoch {epoch + 1} step {step}/{total} loss {loss_sum.item() / loss_n:.3f} '
                    f'{n_seen / (time.time() - t_ep):.0f} img/s')
                if spent() + post_s * rate > budget_usd:
                    stopped_early = f'budget guard mid-epoch {epoch + 1}'
                    break
            if max_steps and step >= max_steps:
                break
        ep_s = time.time() - t_ep
        accs = {}
        for which, net in (('raw', model), ('ema', ema.module)):
            zv = logits(net, va, enhance=False, tta=False)
            accs[which] = float((zv.argmax(1) == y[va]).mean())
        which = max(accs, key=accs.get)
        history.append(dict(epoch=epoch + 1, train_loss=round(loss_sum.item() / max(1, loss_n), 4), val_top1_raw=accs['raw'],
                            val_top1_ema=accs['ema'], train_sec=round(ep_s), img_per_s=round(n_seen / ep_s, 1),
                            spent_usd=round(spent(), 2)))
        log(f'epoch {epoch + 1}/{epochs} done in {ep_s / 60:.1f} min | val top-1 raw {accs["raw"]:.4f} ema {accs["ema"]:.4f}')
        if accs[which] > best['acc']:
            best = dict(acc=accs[which], epoch=epoch + 1, which=which)
            best_sd = copy.deepcopy((model if which == 'raw' else ema.module).state_dict())
            torch.save(best_sd, f'{RUN}/best.pt')
        if not smoke:
            torch.save(dict(model=model.state_dict(), ema=ema.module.state_dict(), opt=opt.state_dict(),
                            scaler=scaler.state_dict(), epoch=epoch, step=step, best=best, history=history, C=C,
                            epochs=epochs), ck_path)
            vol.commit()
        if stopped_early:
            break
        if epoch + 1 < epochs and spent() + (ep_s + 60 + post_s) * rate > budget_usd:
            stopped_early = f'budget guard after epoch {epoch + 1}'
            break
    if stopped_early:
        log('STOPPED EARLY:', stopped_early)
    if best_sd is None:
        best_sd = torch.load(f'{RUN}/best.pt', map_location='cuda', weights_only=False)
    t_train_end = time.time()

    # ---------------- evaluation (app pipeline: CROP_PAD crop, illumination fix, flip TTA) ----------------
    model.load_state_dict(best_sd)
    net = model.eval()
    yv, yt_ = y[va], y[te]
    zv = logits(net, va)
    zt = logits(net, te)
    zt_notta = logits(net, te, tta=False)
    zt_noenh = logits(net, te, enhance=False)
    log('logits done')

    def softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def fit_temperature(z, labels):
        lt, yt = torch.tensor(z, dtype=torch.float32), torch.tensor(labels)
        logT = torch.zeros(1, requires_grad=True)
        o = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

        def closure():
            o.zero_grad()
            l = F.cross_entropy(lt / logT.exp(), yt)
            l.backward()
            return l
        o.step(closure)
        return float(logT.exp())

    def pick_threshold(conf, correct, target, floor=0.30):
        for t in np.linspace(floor, 0.99, 70):
            m = conf >= t
            if m.sum() >= 20 and correct[m].mean() >= target:
                return float(t)
        return 0.9

    def ece(P, labels, bins=15):
        conf, pred = P.max(1), P.argmax(1)
        e = 0.0
        for lo in np.linspace(0, 1, bins + 1)[:-1]:
            m = (conf > lo) & (conf <= lo + 1 / bins)
            if m.any():
                e += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
        return float(e)

    make_names = sorted({c['make'] for c in classes})
    M_make = np.zeros((C, len(make_names)))
    M_make[np.arange(C), [make_names.index(c['make']) for c in classes]] = 1
    model_keys = sorted({(c['make'], c['base_model']) for c in classes})
    M_model = np.zeros((C, len(model_keys)))
    M_model[np.arange(C), [model_keys.index((c['make'], c['base_model'])) for c in classes]] = 1
    make_of, model_of = M_make.argmax(1), M_model.argmax(1)

    def cls_metrics(P, labels):
        pred = P.argmax(1)
        top5 = np.argsort(-P, 1)[:, :5]
        tp = np.bincount(labels[pred == labels], minlength=C).astype(float)
        n_true, n_pred = np.bincount(labels, minlength=C).astype(float), np.bincount(pred, minlength=C).astype(float)
        present = n_true > 0
        rec = np.divide(tp, n_true, out=np.zeros(C), where=n_true > 0)
        prec = np.divide(tp, n_pred, out=np.zeros(C), where=n_pred > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(C), where=(prec + rec) > 0)
        return dict(n=int(len(labels)), top1=float((pred == labels).mean()), top5=float((top5 == labels[:, None]).any(1).mean()),
                    macro_f1=float(f1[present].mean()), balanced_acc=float(rec[present].mean()),
                    make_acc=float(((P @ M_make).argmax(1) == make_of[labels]).mean()),
                    model_acc_any_generation=float(((P @ M_model).argmax(1) == model_of[labels]).mean()))

    T = fit_temperature(zv, yv)
    Pv, Pt = softmax(zv / T), softmax(zt / T)
    tau = pick_threshold(Pv.max(1), Pv.argmax(1) == yv, 0.99)
    MPv = Pv @ M_make
    tau_make = pick_threshold(MPv.max(1), MPv.argmax(1) == make_of[yv], 0.98)
    test = cls_metrics(Pt, yt_)
    conf, ok = Pt.max(1), Pt.argmax(1) == yt_
    MPt = Pt @ M_make
    status = np.where(conf >= tau, 'confident', np.where(MPt.max(1) >= tau_make, 'uncertain_model', 'low_confidence'))
    breakdown = {s: dict(share=float((status == s).mean()), model_acc=float(ok[status == s].mean()) if (status == s).any() else None,
                         make_acc=float((MPt.argmax(1) == make_of[yt_])[status == s].mean()) if (status == s).any() else None)
                 for s in ('confident', 'uncertain_model', 'low_confidence')}
    thr = [dict(threshold=round(t, 1), coverage=float((conf >= t).mean()), accuracy=float(ok[conf >= t].mean()) if (conf >= t).any() else None)
           for t in np.arange(0, 1, 0.1)]
    eras = [('before 1990', 0, 1989), ('1990-1999', 1990, 1999), ('2000-2009', 2000, 2009), ('2010-2019', 2010, 2019),
            ('2020 and newer', 2020, 3000), ('2023 and newer', 2023, 3000)]
    sy = np.array([classes[c]['start_year'] for c in yt_])
    by_era = [dict(era=n, n=int(((sy >= a) & (sy <= b)).sum()), **{k: v for k, v in cls_metrics(Pt[(sy >= a) & (sy <= b)], yt_[(sy >= a) & (sy <= b)]).items() if k in ('top1', 'top5', 'make_acc', 'model_acc_any_generation')})
              for n, a, b in eras if ((sy >= a) & (sy <= b)).any()]
    srcs = np.array([classes[c]['source'] for c in yt_])
    by_source = {s: cls_metrics(Pt[srcs == s], yt_[srcs == s]) for s in sorted(set(srcs))}
    pred = Pt.argmax(1)
    pairs = collections.Counter((int(a), int(b)) for a, b in zip(yt_, pred) if a != b)
    confusions = [dict(true=classes[a]['name'], pred=classes[b]['name'], count=n, same_model=classes[a]['base_model'] == classes[b]['base_model'] and classes[a]['make'] == classes[b]['make'])
                  for (a, b), n in pairs.most_common(25)]
    per_class = []
    for c in classes:
        m = yt_ == c['id']
        per_class.append(dict(id=c['id'], n_train=c['n_train'], n_test=int(m.sum()), acc=float(ok[m].mean()) if m.any() else None))
    ablation = dict(full_pipeline=test['top1'], without_flip_tta=float((zt_notta.argmax(1) == yt_).mean()),
                    without_illumination_fix=float((zt_noenh.argmax(1) == yt_).mean()),
                    best_weights=best['which'], val_top1_best=best['acc'])
    calibration = dict(T=T, ece_before=ece(softmax(zt), yt_), ece_after=ece(Pt, yt_),
                       nll_before=float(-np.log(softmax(zt)[np.arange(len(yt_)), yt_] + 1e-12).mean()),
                       nll_after=float(-np.log(Pt[np.arange(len(yt_)), yt_] + 1e-12).mean()))
    log(f'TEST top-1 {test["top1"]:.4f} top-5 {test["top5"]:.4f} make {test["make_acc"]:.4f} | T {T:.3f} tau {tau:.2f} tau_make {tau_make:.2f}')

    # ---------------- export: checkpoint + ONNX FP32 -> INT8 (dynamic quantisation of MatMul/Gemm) ----------------
    A = f'{OUT}/artifacts'
    sd_cpu = {k: v.detach().float().cpu() for k, v in best_sd.items()}
    torch.save(dict(state_dict=sd_cpu, model_name=MODEL_NAME, num_classes=C, img_size=IMG,
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), f'{A}/classifier_best.pth')
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    cpu_m = timm.create_model(MODEL_NAME, pretrained=False, num_classes=C)
    cpu_m.load_state_dict(sd_cpu)
    cpu_m.eval()
    x1 = torch.randn(1, 3, IMG, IMG)
    f32, i8 = f'{RUN}/classifier_fp32.onnx', f'{A}/classifier_int8.onnx'
    torch.onnx.export(cpu_m, x1, f32, input_names=['input'], output_names=['logits'], dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
                      opset_version=17, dynamo=False)
    quantize_dynamic(f32, i8, weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul', 'Gemm'])
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    s32 = ort.InferenceSession(f32, so, providers=['CPUExecutionProvider'])
    s8 = ort.InferenceSession(i8, so, providers=['CPUExecutionProvider'])
    sub = te[:200 if smoke else 600]
    X = np.stack([load_eval(int(i), True) for i in sub]).astype(np.float32) / 255
    X = ((X - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)).transpose(0, 3, 1, 2).copy()
    torch.set_num_threads(2)
    with torch.no_grad():
        z_t = np.concatenate([cpu_m(torch.from_numpy(X[k:k + 50])).numpy() for k in range(0, len(X), 50)])
    z_32 = np.concatenate([s32.run(None, {'input': X[k:k + 50]})[0] for k in range(0, len(X), 50)])
    z_8 = np.concatenate([s8.run(None, {'input': X[k:k + 50]})[0] for k in range(0, len(X), 50)])

    def ms(fn, n=15):
        fn()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t) / n * 1000

    x1n = X[:1]
    with torch.no_grad():
        ms_t = ms(lambda: cpu_m(torch.from_numpy(x1n)))
    ys = y[sub]
    optimisation = [
        dict(variant='PyTorch FP32 (CPU, 2 threads)', size_mb=round(os.path.getsize(f'{A}/classifier_best.pth') / 1e6, 1), ms=round(ms_t, 1),
             top1=float((z_t.argmax(1) == ys).mean())),
        dict(variant='ONNX Runtime FP32 (CPU, 2 threads)', size_mb=round(os.path.getsize(f32) / 1e6, 1), ms=round(ms(lambda: s32.run(None, {'input': x1n})), 1),
             top1=float((z_32.argmax(1) == ys).mean())),
        dict(variant='ONNX Runtime INT8 (CPU, 2 threads)', size_mb=round(os.path.getsize(i8) / 1e6, 1), ms=round(ms(lambda: s8.run(None, {'input': x1n})), 1),
             top1=float((z_8.argmax(1) == ys).mean()), agreement_with_fp32=float((z_8.argmax(1) == z_t.argmax(1)).mean())),
    ]
    log('optimisation', optimisation)

    # ---------------- config, class list, samples, failures, metrics ----------------
    config = dict(model_name=MODEL_NAME, num_classes=C, img_size=IMG, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                  crop_pad=CROP_PAD, T=T, alpha=0.0, T2=1.0, tau=tau, tau_make=tau_make, det_conf=0.25, yolo_imgsz=640,
                  enhance=True, dark_thr=80, bright_thr=200, tta=True)
    json.dump(config, open(f'{A}/config.json', 'w'), indent=2)
    json.dump(classes, open(f'{A}/class_meta.json', 'w'), indent=1)

    def save_crop(i, path):
        r = recs[i]
        open(path, 'wb').write(shards[r['s']][r['o']:r['o'] + r['n']].tobytes())
        return dict(file=os.path.basename(path), author=r['author'], licence=r['lic'], source=r['url'])

    rng = np.random.default_rng(7)
    cand = [int(i) for i in rng.permutation(te) if classes[y[i]]['start_year'] >= 2020 and recs[i]['src'] == 'commons-hf']
    samples, seen = [], set()
    for i in cand:
        c = classes[y[i]]
        if c['make'] in seen:
            continue
        seen.add(c['make'])
        k = len(samples)
        samples.append(dict(**save_crop(i, f'{OUT}/samples/recent_{k:02d}.jpg'), label=c['name'], class_id=c['id']))
        if len(samples) == 12:
            break
    wrong = np.where(~ok)[0]
    wrong = wrong[np.argsort(-conf[wrong])][:16]
    failures = []
    for k, w in enumerate(wrong):
        i = int(te[w])
        failures.append(dict(**save_crop(i, f'{OUT}/failures/fail_{k:02d}.jpg'), true=classes[yt_[w]]['name'], pred=classes[pred[w]]['name'],
                             confidence=float(conf[w])))
    json.dump(dict(samples=samples, failures=failures), open(f'{OUT}/credits.json', 'w'), indent=1)

    n_src = collections.Counter(r['src'] for r, c in zip(recs, y) if c >= 0)
    metrics = dict(
        dataset=dict(name='US car models 1948-2026 (Wikimedia Commons) + Stanford Cars train images for 14 exotic makes',
                     hf_repo=HF_REPO, hf_revision=HF_REV, images_used=int((y >= 0).sum()), images_by_source=dict(n_src),
                     classes=C, makes=len(make_names), models=len(model_keys), min_photos_per_class=MIN_PER_CLASS,
                     split=dict(train=int(len(tr)), val=int((split == 1).sum()), test=int((split == 2).sum())),
                     split_rule='80/10/10 per class, grouped by photographer and day; random split where one photographer dominates',
                     random_split_classes=fallback,
                     years=dict(first=min(c['start_year'] for c in classes if c['start_year']), last_generation_start=max(c['start_year'] for c in classes))),
        training=dict(gpu=gpu, device=torch.cuda.get_device_name(), precision=str(amp_dt).split('.')[-1], epochs_planned=epochs,
                      epochs_run=len(history), stopped_early=stopped_early or None, batch=batch, lr=lr, head_lr_mult=5, weight_decay=0.05,
                      label_smoothing=0.1, ema_decay=0.9995, img_size=IMG, train_minutes=round((t_train_end - T0) / 60, 1),
                      total_minutes=round((time.time() - T0) / 60, 1), cost_usd_this_run=round((time.time() - T0) * rate, 2),
                      cost_usd_total_estimate=round(spent(), 2), history=history),
        test=test, by_era=by_era, by_source=by_source, calibration=calibration,
        confidence=dict(tau=tau, tau_make=tau_make, target_precision=0.99, target_make_precision=0.98, decision_breakdown=breakdown,
                        threshold_table=thr),
        ablation=ablation, optimisation=optimisation, top_confusions=confusions, per_class=per_class,
        smoke_test=smoke,
    )
    json.dump(metrics, open(f'{A}/metrics.json', 'w'), indent=1)
    with gzip.open(f'{OUT}/split_index.json.gz', 'wt') as f:
        json.dump([[r['f'], int(c), int(s)] for r, c, s in zip(recs, y, split) if c >= 0], f)
    vol.commit()
    log('done')
    return dict(test=test, calibration=calibration, confidence=dict(tau=tau, tau_make=tau_make), optimisation=optimisation,
                training={k: metrics['training'][k] for k in ('epochs_run', 'stopped_early', 'train_minutes', 'total_minutes', 'cost_usd_this_run')})


@app.local_entrypoint()
def train(epochs: int = 12, batch: int = 128, lr: float = 4e-4, smoke: bool = False, compile_model: bool = True,
          spent_usd: float = SPENT_BEFORE_TRAIN):
    print(f'GPU {TRAIN_GPU} | hard timeout {TRAIN_TIMEOUT / 3600:.2f} h | worst case ${spent_usd + TRAIN_TIMEOUT * rate_per_s(TRAIN_GPU):.2f}')
    res = train_remote.remote(epochs=epochs, batch=batch, lr=lr, smoke=smoke, compile_model=compile_model, spent_usd=spent_usd,
                             gpu_name=TRAIN_GPU)
    print(json.dumps(res, indent=1))


@app.local_entrypoint()
def fetch(smoke: bool = False, dest: str = ''):
    """Copy run/out (or smoke/out) from the volume into the project (artifacts_modern/ by default)."""
    src = 'smoke/out' if smoke else 'run/out'
    out = pathlib.Path(dest) if dest else ROOT / ('artifacts_modern_smoke' if smoke else 'artifacts_modern_new')
    n = 0
    for e in vol.listdir(src, recursive=True):
        if e.type != modal.volume.FileEntryType.FILE:
            continue
        rel = pathlib.Path(e.path).relative_to(src)
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'wb') as f:
            for chunk in vol.read_file(e.path):
                f.write(chunk)
        n += 1
    print(f'{n} files -> {out}')
