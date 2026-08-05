#!/usr/bin/env python
"""Generate every paper figure from SAVED results - no pipeline run needed.

    python scripts/make_figures.py --cavity DARM
    python scripts/make_figures.py --cavity ALL --run-type Sweep
    python scripts/make_figures.py --table-only
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')          # no display needed; write straight to disk

from ligo_microseism.config import Run_Type, Training_Type, Cavity
from ligo_microseism.registry import Cavity_Registry
from ligo_microseism.results import load_results, load_fit, summary_table
from ligo_microseism.plotting import Blind_Plots, Sweep_Plots, FIR_Channel_Plot


def main():
    p = argparse.ArgumentParser(description='Build figures from saved results')
    p.add_argument('--cavity', default=Cavity)
    p.add_argument('--run-type', default=Run_Type, choices=['Blind', 'Sweep'])
    p.add_argument('--training-type', default=Training_Type, choices=['Noisy', 'Quiet'])
    p.add_argument('--outdir', default='figures')
    p.add_argument('--results-dir', default='results')
    p.add_argument('--table-only', action='store_true')
    args = p.parse_args()

    cavities = (list(Cavity_Registry) if args.cavity.upper() == 'ALL'
                else [args.cavity])

    print('Summary table:')
    summary_table(cavities, args.run_type, args.training_type, args.results_dir)
    if args.table_only:
        return

    for cav in cavities:
        try:
            results, meta = load_results(cav, args.run_type,
                                         args.training_type, args.results_dir)
        except FileNotFoundError:
            print(f'  skipping {cav}: no saved results')
            continue
        print(f'\n{cav}  (run {meta["created"]})')

        # the fitted filter: one summary plot per fit, redrawn from the saved
        # weights - no refit needed
        fit = load_fit(cav, args.run_type, args.training_type, args.results_dir)
        if 'weights' in fit:
            Path(args.outdir).mkdir(parents=True, exist_ok=True)
            dc = FIR_Channel_Plot(fit['weights'], fit['X_Std'], fit['y_Std'],
                                  Cavity_Registry[cav]['CPS_Channels'],
                                  save_path=f'{args.outdir}/{cav}_fir_taps.pdf',
                                  show=False)
            dc.to_csv(f'{args.results_dir}/{cav}_dc_gains.csv', index=False)
            print(dc.to_string(index=False))

        if args.run_type == 'Blind':
            Blind_Plots(results, cav, save_dir=args.outdir, show=False)
        else:
            Sweep_Plots(results, cav, Cavity_Registry[cav]['CPS_Channels'],
                        training_type=args.training_type,
                        save_dir=args.outdir, show=False)


if __name__ == '__main__':
    main()
