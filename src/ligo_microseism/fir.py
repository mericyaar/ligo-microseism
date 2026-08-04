import numpy as np
from scipy.linalg import solve, LinAlgError
from ligo_microseism.config import TAPS, Causality_gap

def FIR_Normal_Eq(Xz, yz, K=TAPS, gap=Causality_gap, chunk=6000):
    lags = list(range(gap, K + gap))
    lags_arr = np.array(lags)
    n, S = Xz.shape
    D = S * len(lags)
    lo, hi = min(lags), max(lags)
    start, stop = max(0, hi), n + min(0, lo)
    A = np.zeros((D, D))
    b = np.zeros(D)
    
    for s in range(start, stop, chunk):
        e = min(s + chunk, stop)
        row_idx = np.arange(s, e)[:, None]
        col_idx = lags_arr[None, :]
        time_idx = row_idx - col_idx

        Phi_3D = Xz[time_idx]
        Phi_3D_Trans = Phi_3D.transpose(0, 2, 1)
        Phi = Phi_3D_Trans.reshape(e - s, D)

        for block_start in range(0, e - s, 512):
            block_end = min(block_start + 512, e - s)
            Phi_block = Phi[block_start:block_end]
            A += Phi_block.T @ Phi_block
            b += Phi_block.T @ yz[s + block_start:s + block_end]

    return A, b, stop - start

def FIR_Prediction(Xz, w, K=TAPS, gap = Causality_gap, chunk=1000):
    lags = list(range(gap, K + gap))
    n, C = Xz.shape
    D = C * len(lags)
    lo, hi = min(lags), max(lags)
    start, stop = max(0, hi), n + min(0, lo)

    yhat = np.full(n, np.nan)
    lags_arr = np.array(lags)
    for s in range(start, stop, chunk):
        e = min(s + chunk, stop)
        row_idx = np.arange(s, e)[:, None]
        lag_idx = lags_arr[None, :]
        time_idx = row_idx - lag_idx
        Phi = Xz[time_idx].transpose(0, 2, 1).reshape(e - s, D)
        yhat[s:e] = Phi @ w
    return yhat

def _safe_solve(M, rhs):
    try:
        return solve(M, rhs, assume_a="pos", check_finite=False)
    except LinAlgError:
        print('FIR is not solved in closed form!')
        return np.linalg.lstsq(M, rhs, rcond=None)[0]