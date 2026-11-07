"""Run notebook/P5_Evaluation_Remaining.ipynb on a Modal GPU instead of Colab.

    .venv/bin/modal run modal_eval/run_eval.py                 # upload models (first time), run, download results
    .venv/bin/modal run modal_eval/run_eval.py --no-download   # leave results on the volume
    P5_GPU=A10G .venv/bin/modal run modal_eval/run_eval.py     # faster GPU (latency numbers then refer to an A10G)

What it does
  * image: Python 3.11, torch 2.8 (CUDA), ultralytics, timm, onnx/onnxruntime, scikit-learn 1.6.1 (the version that pickled
    classical_lr.joblib), Jupyter runtime, and the Stanford Cars GitHub mirror cloned at /content/StanfordCars-Dataset
    (exactly the path the notebook expects, so its `git clone` cell is skipped).
  * volume `p5-eval` is mounted at /content/P5_vehicle_recognition, which is the notebook's own non-Colab OUT path, so
    artifacts/, plots/ and eval_cache/ persist between runs. A re-run after a crash reuses the cached predictions and the
    finished robustness conditions, as on Colab.
  * the notebook runs unmodified except that the `!pip install` line is skipped (dependencies are baked into the image).
  * after the run, P5_artifacts.zip, P5_plots.zip and the executed notebook are downloaded to modal_runs/<timestamp>/.
"""
import os
import pathlib
import time

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB_PATH = ROOT / 'notebook' / 'P5_Evaluation_Remaining.ipynb'
OUT = '/content/P5_vehicle_recognition'
GPU = os.environ.get('P5_GPU', 'T4')
EXECUTED = 'P5_Evaluation_Remaining.executed.ipynb'

app = modal.App('p5-evaluation')
vol = modal.Volume.from_name('p5-eval', create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('git', 'libgl1', 'libglib2.0-0')
    .pip_install('torch==2.8.0', 'torchvision==0.23.0')
    .pip_install(
        'ultralytics', 'timm', 'onnx', 'onnxruntime', 'onnxscript', 'scikit-image', 'scikit-learn==1.6.1', 'scipy',
        'pandas', 'matplotlib', 'opencv-python-headless', 'joblib', 'tqdm', 'Pillow',
        'nbclient', 'nbformat', 'ipykernel',
    )
    .run_commands('git clone --depth 1 https://github.com/jhpohovey/StanfordCars-Dataset.git /content/StanfordCars-Dataset')
)


CPUS = 8

# Inserted right after the notebook's config cell. Only affects parallelism (never results):
#  * os.cpu_count() in a container reports the host's CPUs; cap it to the CPUs reserved for this function
#  * if /dev/shm is small, DataLoader workers can crash on shared-memory tensors -> load in-process instead
MODAL_SETUP = f'''# [modal] runtime adjustments (parallelism only)
import os, shutil
os.cpu_count = lambda: {CPUS}
_shm = shutil.disk_usage('/dev/shm').total / 1e9 if os.path.exists('/dev/shm') else 0
if _shm < 2:
    NW = 0
print(f'[modal] /dev/shm {{_shm:.1f}} GB -> DataLoader workers = {{NW}} | Pool size = {{os.cpu_count()}} | device = {{DEVICE}}')
'''


def _patch(nb):
    """Skip `!pip install` (deps are in the image) and add MODAL_SETUP after the config cell. Nothing else changes."""
    import nbformat
    cells = []
    for c in nb.cells:
        if c.cell_type == 'code' and '!pip' in c.source:
            c.source = '\n'.join(('# [modal] ' + l if l.strip().startswith('!pip') else l) for l in c.source.split('\n'))
        cells.append(c)
        if c.cell_type == 'code' and 'CFG = dict(' in c.source and 'NUM_WORKERS' in c.source:
            cells.append(nbformat.v4.new_code_cell(MODAL_SETUP))
    nb.cells = cells
    return nb


@app.function(image=image, gpu=GPU, cpu=float(CPUS), memory=32768, timeout=4 * 60 * 60, volumes={OUT: vol})
def execute(nb_json: str) -> dict:
    import threading
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = _patch(nbformat.reads(nb_json, as_version=4))
    os.makedirs('/content', exist_ok=True)
    t0, starts = time.time(), {}
    stamp = lambda: time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))
    title = lambda c: next((l.strip() for l in c.source.split('\n') if l.strip()), '')[:90]

    # commit the volume from a plain thread every minute (caches survive a crash); never from nbclient's event loop
    stop = threading.Event()
    def committer():
        while not stop.wait(60):
            try: vol.commit()
            except Exception as e: print('[modal] volume commit failed:', e, flush=True)
    threading.Thread(target=committer, daemon=True).start()

    def save():
        nbformat.write(nb, f'{OUT}/{EXECUTED}')

    def on_start(cell, cell_index, **_):
        if cell.cell_type == 'code':
            starts[cell_index] = time.time()
            print(f'[{stamp()}] ▶ cell {cell_index}: {title(cell)}', flush=True)

    def on_done(cell, cell_index, **_):
        if cell.cell_type != 'code':
            return
        text = []
        for o in cell.get('outputs', []):
            if o.get('output_type') == 'stream':
                text.append(o.get('text', ''))
            elif o.get('output_type') in ('execute_result', 'display_data'):
                text.append(o.get('data', {}).get('text/plain', ''))
            elif o.get('output_type') == 'error':
                text.append(f"{o.get('ename')}: {o.get('evalue')}")
        lines = [l for l in '\n'.join(text).replace('\r', '\n').split('\n') if l.strip() and '%|' not in l]
        for l in lines[-12:]:
            print('      ' + l[:200], flush=True)
        print(f'[{stamp()}] ✓ cell {cell_index} done in {time.time() - starts.get(cell_index, time.time()):.0f}s', flush=True)
        save()

    client = NotebookClient(nb, timeout=None, kernel_name='python3', resources={'metadata': {'path': '/content'}},
                            on_cell_start=on_start, on_cell_executed=on_done)
    try:
        client.execute()
        ok, err = True, None
    except CellExecutionError as e:
        ok, err = False, str(e)[-4000:]
        print('\n✗ notebook stopped with an error:\n' + err, flush=True)
    save()
    stop.set()
    vol.commit()
    files = sorted(os.listdir(f'{OUT}/artifacts')) if os.path.isdir(f'{OUT}/artifacts') else []
    return dict(ok=ok, error=err, minutes=round((time.time() - t0) / 60, 1), artifacts=files)


def _exists(path: str) -> bool:
    try:
        parent, name = path.rsplit('/', 1)
        return any(e.path.rstrip('/').endswith(name) for e in vol.listdir(parent or '/'))
    except Exception:
        return False


@app.local_entrypoint()
def main(download: bool = True):
    art = ROOT / 'artifacts'
    # first run only: trained models + the training notebook's original metrics.json (the notebook extends it)
    uploads = {f'/artifacts/{f}': art / f for f in ['yolo_car_best.pt', 'classifier_best.pth', 'classical_lr.joblib']}
    uploads['/artifacts/metrics.json'] = art / 'metrics.colab.json'
    todo = {dst: src for dst, src in uploads.items() if not _exists(dst)}
    if todo:
        print('Uploading to volume p5-eval:', ', '.join(todo))
        with vol.batch_upload() as b:
            for dst, src in todo.items():
                b.put_file(str(src), dst)
    else:
        print('Models already on volume p5-eval (re-run: cached predictions/robustness are reused).')

    print(f'Running the notebook on a Modal {GPU} GPU…')
    res = execute.remote(NB_PATH.read_text())
    print('\nResult:', {k: v for k, v in res.items() if k != 'error'})
    if res.get('error'):
        print('Error tail:\n', res['error'][-1500:])

    if download:
        dest = ROOT / 'modal_runs' / time.strftime('%Y%m%d-%H%M%S')
        dest.mkdir(parents=True, exist_ok=True)
        for name in ['P5_artifacts.zip', 'P5_plots.zip', EXECUTED]:
            try:
                with open(dest / name, 'wb') as f:
                    for chunk in vol.read_file(name):
                        f.write(chunk)
                print(f'downloaded {name} ({(dest / name).stat().st_size / 1e6:.1f} MB)')
            except Exception as e:
                (dest / name).unlink(missing_ok=True)
                print(f'not available: {name} ({type(e).__name__})')
        if (dest / 'P5_artifacts.zip').exists():
            print(f'\nNext: .venv/bin/python scripts/import_colab_results.py --artifacts {dest}/P5_artifacts.zip --plots {dest}/P5_plots.zip')
