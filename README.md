# FIR Microseismic Noise Mitigation (LIGO)

This repo contains the code to learn and suppress microseismic motion from monitoring seismometers using a causal Finite Impulse Response (FIR) engine. It is built as a Python library (`ligo_microseism`) to facilitate stable fitting, leave-one-folder-out (LOFO) cross-validation, and coherence evaluation. This README will guide you through data preparation, training, and evaluation.

### Data preparation
Please use the provided `.txt` and `.npy` files. To regenerate the inputs from LIGO archives, refer to the exact GPS folder timestamps and optical coefficients documented in our reproducibility contract: [data/MANIFEST.md](data/MANIFEST.md).

### Training (FIR LOFO Fitting)
The FIR filter weights are solved using a chunked, Cholesky-based Ridge regression. The pipeline is orchestrated via thin CLI scripts that read defaults from `src/ligo_microseism/config.py` but allow dynamic command-line overrides. Start a blind evaluation or hyperparameter sweep by running:
```bash
python scripts/run_fir.py --cavity DARM --run-type Blind
python scripts/run_fir.py --cavity PRCL --run-type Sweep
```

### Evaluation
Evaluation is built directly into the pipeline execution. The script automatically evaluates the model against blind datasets and generate diagnostic figures (ASD, time-series, and coherence).


## Future Work
The codebase is ready to run parallel trainings and sweeps via the CLI. We are currently migrating our self-test battery into tests/ to establish a strict pytest harness for causality (using the causality_gap horizon) and future-leakage prevention. We also plan to automate paper-ready tables from results/*.json and connect the repository to Zenodo for publication.