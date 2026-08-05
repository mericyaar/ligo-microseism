"""Two artefacts per run:

  results/<cav>_<runtype>_<trainingtype>.json  - scalars + metadata 
  results/<cav>_<runtype>_<trainingtype>.npz   - arrays (ASDs, time series,
                                                 weights) for the figures

the JSON stays small enough to read and commit, while
the bulky arrays live in a binary file that .gitignore can drop.
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

RESULTS_DIR = Path('results')

# keys in a per-folder result dict that are arrays (everything else is scalar)
_ARRAY_KEYS = {'yhat', 'y', 'f', 'yhat_ASD', 'y_ASD', 'Res_ASD', 'f_coh',
               'FIR_coh', 'Combo_coh', 'weights', 'Delay', 'Delay_ASD',
               'Delay_Res_ASD'}


def run_tag(cav, run_type, training_type):
    return f'{cav}_{run_type}_{training_type}'


def save_results(results, cav, run_type, training_type, config=None,
                 outdir=RESULTS_DIR, extras=None):
    """Split `results` (nested: group -> folder -> dict) into JSON + NPZ.

    `extras` holds the FIT ITSELF (weights, alpha, z-scoring constants) so the
    filter that produced these numbers is stored WITH them - that is what lets
    make_figures.py redraw the tap plot without refitting.  Arrays go to the
    NPZ under a '__fit__|' prefix; scalars go into the JSON meta.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = run_tag(cav, run_type, training_type)

    scalars, arrays = {}, {}
    for group, folders in results.items():
        if not isinstance(folders, dict):
            continue
        scalars[group] = {}
        for name, d in folders.items():
            s = {}
            for k, v in d.items():
                if k in _ARRAY_KEYS or isinstance(v, np.ndarray):
                    arrays[f'{group}|{name}|{k}'] = np.asarray(v)
                elif isinstance(v, (int, float, np.floating, np.integer)):
                    s[k] = float(v)
                elif isinstance(v, str):
                    s[k] = v
            scalars[group][name] = s

    fit_scalars = {}
    for k, v in (extras or {}).items():
        if isinstance(v, np.ndarray):
            arrays[f'__fit__|{k}|_'] = v
        elif isinstance(v, (int, float, np.floating, np.integer)):
            fit_scalars[k] = float(v)
        elif isinstance(v, str):
            fit_scalars[k] = v

    meta = dict(cavity=cav, run_type=run_type, training_type=training_type,
                created=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                config=config or {}, fit=fit_scalars)
    with open(outdir/f'{tag}.json', 'w') as fh:
        json.dump(dict(meta=meta, results=scalars), fh, indent=2)
    np.savez_compressed(outdir/f'{tag}.npz', **arrays)

    n = sum(len(v) for v in scalars.values())
    print(f'  wrote {outdir}/{tag}.json  ({n} folders)')
    print(f'  wrote {outdir}/{tag}.npz   ({len(arrays)} arrays)')
    return outdir/f'{tag}.json'


def load_results(cav, run_type, training_type, outdir=RESULTS_DIR,
                 with_arrays=True):
    """Rebuild the nested dict that the plotting functions expect."""
    outdir = Path(outdir)
    tag = run_tag(cav, run_type, training_type)
    with open(outdir/f'{tag}.json') as fh:
        blob = json.load(fh)
    results = {g: {n: dict(d) for n, d in folders.items()}
               for g, folders in blob['results'].items()}
    if with_arrays:
        with np.load(outdir/f'{tag}.npz') as z:
            for key in z.files:
                group, name, k = key.split('|', 2)
                if group == '__fit__':
                    continue                      # retrieved via load_fit()
                results.setdefault(group, {}).setdefault(name, {})[k] = z[key]
    return results, blob['meta']


def load_fit(cav, run_type, training_type, outdir=RESULTS_DIR):
    """Return the saved fit: weights, alpha, and the z-scoring constants."""
    outdir = Path(outdir)
    tag = run_tag(cav, run_type, training_type)
    with open(outdir/f'{tag}.json') as fh:
        fit = dict(json.load(fh)['meta'].get('fit', {}))
    with np.load(outdir/f'{tag}.npz') as z:
        for key in z.files:
            group, name, _ = key.split('|', 2)
            if group == '__fit__':
                fit[name] = z[key]
    return fit


def summary_table(cavities, run_type, training_type, outdir=RESULTS_DIR):
    """One row per (cavity, group, folder) - the backbone of the paper table."""
    import pandas as pd
    rows = []
    for cav in cavities:
        try:
            res, meta = load_results(cav, run_type, training_type, outdir,
                                     with_arrays=False)
        except FileNotFoundError:
            print(f'  (no results yet for {cav})')
            continue
        for group, folders in res.items():
            for name, d in folders.items():
                rows.append(dict(cavity=cav, group=group, folder=name,
                                 **{k: v for k, v in d.items()
                                    if isinstance(v, float)}))
    df = pd.DataFrame(rows)
    if len(df):
        out = Path(outdir)/f'summary_{run_type}_{training_type}.csv'
        df.to_csv(out, index=False)
        print(f'  wrote {out}  ({len(df)} rows)')
    return df
