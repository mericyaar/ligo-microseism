#!/usr/bin/env python

import argparse
import matplotlib
#matplotlib.use('MacOSX')

# Import your default controls and parameters from the config
from ligo_microseism.config import (
    Run_Type, Training_Type, Cavity, TAPS, 
    Causality_gap, Bandpass_Blind, Bandpass_Training, 
    FREQ_BAND, FS, Alphas
)
from ligo_microseism.registry import Cavity_Registry
from ligo_microseism.loader import build_library
from ligo_microseism.pipeline import LOFO, Blind_Eval, Sweep_Cavity
from ligo_microseism.plotting import Blind_Plots, Sweep_Plots
from ligo_microseism.results import save_results

cfg = dict(TAPS=TAPS, Causality_gap=Causality_gap, FREQ_BAND=list(FREQ_BAND),
           FS=FS, Alphas=list(Alphas), Bandpass_Training=Bandpass_Training)
def main():
    parser = argparse.ArgumentParser(description="LIGO Microseism FIR Processing Pipeline")
    
    # Set the defaults to the imported config variables
    parser.add_argument('--cavity', type=str, default=Cavity, choices=Cavity_Registry.keys(), 
                        help=f"Which cavity to run (default from config: {Cavity})")
    parser.add_argument('--run-type', type=str, default=Run_Type, choices=['Blind', 'Sweep'], 
                        help=f"Blind or Sweep evaluation (default from config: {Run_Type})")
    parser.add_argument('--training-type', type=str, default=Training_Type, choices=['Noisy', 'Quiet'], 
                        help=f"Which folders to train on (default from config: {Training_Type})")
    
    parser.add_argument('--eval-other', action='store_true', help="If Sweep, evaluate the other type of data not in training")
    parser.add_argument('--no-delay', action='store_false', dest='run_delay', help="Disable the delay comparison model")
    parser.add_argument('--plot', default = False, action='store_true', help="Generate and display plots at the end of the run")
    
    args = parser.parse_args()

    print(":"*75)
    print("LIGO Microseism Pipeline")
    # Print the evaluated args so you know if the CLI successfully overrode the config
    print(f"Cavity: {args.cavity} | Run Type: {args.run_type} | Training: {args.training_type}")
    print(f"Frequency: {FS} Hz | Frequency Band: {FREQ_BAND}")
    print(f"Taps: {TAPS} | Causality Gap: {Causality_gap} | Bandpass Training: {Bandpass_Training}")
    print(":"*75)

    # 1. Fetch the cavity definition
    cav = Cavity_Registry[args.cavity]
    
    # 2. Build the data library using the evaluated arguments
    data_library = build_library(
        cav=cav, 
        Run_Type=args.run_type, 
        Training_Type=args.training_type, 
        Eval_Other=args.eval_other
    )

    # 3. Route to the requested pipeline engine
    if args.run_type == 'Blind':
        print("\n--- Starting LOFO Training ---")
        final_w, final_alpha, X_Mean, X_Std, y_Mean, y_Std = LOFO(data_library)
        
        print("\n--- Starting Blind Evaluation ---")
        results = Blind_Eval(
            data_library, final_w, final_alpha, 
            X_Mean, X_Std, y_Mean, y_Std, 
            run_delay=args.run_delay
        )
        save_results(results, args.cavity, 'Blind', args.training_type, config=cfg,
             extras=dict(weights=final_w, alpha=final_alpha,
                         X_Mean=X_Mean, X_Std=X_Std,
                         y_Mean=y_Mean, y_Std=y_Std))
        if args.plot:
            Blind_Plots(results, args.cavity)

            
    elif args.run_type == 'Sweep':
        print("\n--- Starting Sweep Evaluation ---")
        sweep_results = Sweep_Cavity(
            data_library, 
            eval_other=args.eval_other, 
            run_delay=args.run_delay
        )
        save_results(results, args.cavity, 'Sweep', args.training_type, config=cfg)
        if args.plot:
            Sweep_Plots(
                sweep_results, 
                args.cavity, 
                cav['CPS_Channels'], 
                suppression_plot='All' if args.eval_other else args.training_type,
                training_type=args.training_type
            )

if __name__ == '__main__':
    main()