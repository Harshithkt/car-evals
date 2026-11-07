"""Import the zips downloaded from Colab (P5_Evaluation_Remaining.ipynb → section G.2) into this project.

    python scripts/import_colab_results.py                       # newest P5_artifacts*.zip / P5_plots*.zip in ~/Downloads
    python scripts/import_colab_results.py --artifacts a.zip --plots p.zip
    python scripts/import_colab_results.py --dry-run             # only show what would change

Safety:
  * the current artifacts/metrics.json, config.json and class_meta.json are backed up to artifacts/backup-<timestamp>/
  * zip entries with absolute paths or '..' are rejected
  * inference.py / classical_cv.py are NOT overwritten if they differ from the deployed copies (saved as *.colab.py instead),
    so the app's ML behaviour never changes silently
  * files identical to the current ones are skipped
"""
import argparse, glob, hashlib, json, os, shutil, sys, time, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART, PLOTS = os.path.join(ROOT, 'artifacts'), os.path.join(ROOT, 'plots')
PROTECTED = {'inference.py', 'classical_cv.py'}
EXPECTED_KEYS = ['classical', 'detection', 'detection_ultralytics_test', 'classifier_training', 'calibration_params', 'ablation',
                 'approach_comparison', 'final_system_test', 'similar_class_analysis', 'robustness', 'slices', 'ece', 'confidence',
                 'optimization', 'failure_causes', 'failure_examples', 'end_to_end_ms_per_image', 'requirements_summary']
EXPECTED_PLOTS = ['approach_comparison', 'training_curves', 'detection_examples', 'robustness_bars', 'robustness_examples', 'slices',
                  'reliability', 'threshold_analysis', 'failure_causes', 'failure_gallery', 'error_hierarchy', 'confused_pair', 'optimization']


def newest(pattern):
    hits = sorted(glob.glob(os.path.expanduser(pattern)), key=os.path.getmtime, reverse=True)
    return hits[0] if hits else None


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_members(z):
    """Flatten a zip (it may contain a top-level folder) and reject unsafe names."""
    out = []
    for m in z.infolist():
        if m.is_dir():
            continue
        name = m.filename.replace('\\', '/')
        if name.startswith('/') or '..' in name.split('/') or name.startswith('__MACOSX/'):
            print(f'  ! skipping unsafe or junk entry: {name}')
            continue
        out.append((m, os.path.basename(name)))
    return out


def extract(zip_path, dest, dry, kind):
    changed = []
    with zipfile.ZipFile(zip_path) as z:
        for m, base in safe_members(z):
            if not base or base.startswith('.'):
                continue
            target = os.path.join(dest, base)
            tmp = target + '.importing'
            with z.open(m) as src, open(tmp, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            if os.path.exists(target) and sha(tmp) == sha(target):
                os.remove(tmp)
                continue
            if kind == 'artifacts' and base in PROTECTED and os.path.exists(target):
                alt = target.replace('.py', '.colab.py')
                print(f'  ! {base} differs from the deployed copy; NOT replacing it. Saved as {os.path.basename(alt)} for review.')
                if dry: os.remove(tmp)
                else: os.replace(tmp, alt)
                continue
            changed.append(base)
            if dry: os.remove(tmp)
            else: os.replace(tmp, target)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifacts', default=None)
    ap.add_argument('--plots', default=None)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    art_zip = a.artifacts or newest('~/Downloads/P5_artifacts*.zip')
    plot_zip = a.plots or newest('~/Downloads/P5_plots*.zip')
    if not art_zip and not plot_zip:
        sys.exit('No P5_artifacts*.zip / P5_plots*.zip found. Pass --artifacts / --plots explicitly.')
    for label, p in [('artifacts', art_zip), ('plots', plot_zip)]:
        if p:
            print(f'{label:9s}: {p}  ({time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))}, {os.path.getsize(p) / 1e6:.1f} MB)')

    if not a.dry_run:
        backup = os.path.join(ART, f'backup-{time.strftime("%Y%m%d-%H%M%S")}')
        os.makedirs(backup, exist_ok=True)
        for f in ['metrics.json', 'config.json', 'class_meta.json']:
            if os.path.exists(os.path.join(ART, f)):
                shutil.copy2(os.path.join(ART, f), backup)
        print(f'backup   : {os.path.relpath(backup, ROOT)}/')

    if art_zip:
        ch = extract(art_zip, ART, a.dry_run, 'artifacts')
        print(f'\nartifacts {"would change" if a.dry_run else "updated"}: {", ".join(ch) or "nothing (all identical)"}')
    if plot_zip:
        ch = extract(plot_zip, PLOTS, a.dry_run, 'plots')
        print(f'plots     {"would change" if a.dry_run else "updated"}: {", ".join(ch) or "nothing (all identical)"}')

    # ---- report
    m = json.load(open(os.path.join(ART, 'metrics.json')))
    cfg = json.load(open(os.path.join(ART, 'config.json')))
    have_plots = {os.path.splitext(f)[0] for f in os.listdir(PLOTS) if f.endswith('.png')}
    print('\nmetrics.json keys present :', ', '.join(k for k in EXPECTED_KEYS if k in m))
    missing = [k for k in EXPECTED_KEYS if k not in m]
    print('metrics.json keys missing :', ', '.join(missing) or 'none')
    print('plots missing             :', ', '.join(p for p in EXPECTED_PLOTS if p not in have_plots) or 'none')
    print('ONNX INT8 model           :', 'yes' if os.path.exists(os.path.join(ART, 'classifier_int8.onnx')) else 'no')
    print(f"thresholds in config.json: tau={cfg.get('tau')}  tau_make={cfg.get('tau_make')}")
    if 'final_system_test' in m:
        print('final system on test      :', {k: round(v, 4) for k, v in m['final_system_test'].items()})
    print('\nRestart the backend so it loads the new config.json: stop uvicorn (Ctrl+C) and start it again.')


if __name__ == '__main__':
    main()
