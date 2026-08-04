import numpy as np
from pathlib import Path
from scipy.signal import decimate, sosfilt, sosfilt_zi, welch, coherence

from ligo_microseism.config import FS, DURATION, TRIM, _HP_SOS, _LOSS_BP_SOS, FREQ_BAND

def load_sensor(path, filename, fs=FS, duration=DURATION):
    file_path = Path(path) / filename
    raw = np.loadtxt(file_path)
    Org_fs = len(raw) / duration
    q = int(round(Org_fs / fs))
    while q > 1:
        step = min(q, 8)
        raw = decimate(raw, step, ftype='fir', zero_phase=False)
        q //= step
    return raw, Org_fs

def highpass(signal):
    zi = sosfilt_zi(_HP_SOS)
    out, _ = sosfilt(_HP_SOS, signal, axis=0, zi=zi*signal[0])
    return out[TRIM:-TRIM]

def bandpass(signal):
    zi = sosfilt_zi(_LOSS_BP_SOS)
    out, _ = sosfilt(_LOSS_BP_SOS, signal, axis=0, zi=zi*signal[0])
    return out[TRIM:-TRIM]

def compute_ASD(signal, fs=FS, nperseg=4096):
    f, PSD = welch(signal, fs, nperseg=nperseg)
    ASD = np.sqrt(PSD)
    return f, ASD

def CRMS_Inband(signal, fs=FS, nperseg=4096, band=FREQ_BAND):
    f, PSD = welch(signal, fs, nperseg=nperseg)
    m = (f >= band[0]) & (f <= band[1])
    return float(np.sqrt(np.trapezoid(PSD[m], f[m])))
    

def suppression(y, yhat):
    true = CRMS_Inband(y)
    residual = CRMS_Inband(y - yhat)
    return true / residual

def inband_variance_explained(y, yhat):
    true = CRMS_Inband(y)
    residual = CRMS_Inband(y - yhat)
    if true == 0:
        return np.nan
    return 1.0 - (residual / true)**2

def z_score(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    signal_Z = (signal - mean) / std
    return signal_Z, mean, std

def shift_without_wrap(signal, shift):
    signal = np.asarray(signal, dtype=float)
    shifted = np.zeros_like(signal)
    if shift > 0:
        shifted[shift:] = signal[:-shift]
    elif shift < 0:
        shifted[:shift] = signal[-shift:]
    else:
        shifted[:] = signal
    return shifted

def _bandpass_for_delay_fit(signal):
    signal = np.asarray(signal, dtype=float)
    zi = sosfilt_zi(_LOSS_BP_SOS)
    filtered, _ = sosfilt(_LOSS_BP_SOS, signal, axis=0, zi=zi*signal[0])
    if len(filtered) <= 2 * TRIM:
        return filtered
    return filtered[TRIM:-TRIM]

def best_delay_and_gain(combo, y, max_lag=10):
    combo = np.asarray(combo, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(combo), len(y))
    combo = combo[:n]
    y = y[:n]
    max_lag_samples = int(round(max_lag * FS))
    best_shift = None
    best_gain = None
    best_residual_crms = np.inf
    best_prediction = None

    for shift in range(-max_lag_samples, max_lag_samples + 1):
        combo_shifted = shift_without_wrap(combo, shift)
        if shift > 0:
            combo_valid = combo_shifted[shift:]
            y_valid = y[shift:]
        elif shift < 0:
            combo_valid = combo_shifted[:shift]
            y_valid = y[:shift]
        else:
            combo_valid = combo_shifted
            y_valid = y
        if len(combo_valid) < 2:
            continue
            
        combo_fit = _bandpass_for_delay_fit(combo_valid)
        y_fit = _bandpass_for_delay_fit(y_valid)
        fit_n = min(len(combo_fit), len(y_fit))
        combo_fit = combo_fit[:fit_n]
        y_fit = y_fit[:fit_n]
        denominator = np.dot(combo_fit, combo_fit)
        
        if not np.isfinite(denominator) or denominator <= 0:
            continue
            
        gain = float(np.dot(combo_fit, y_fit) / denominator)
        prediction_valid = gain * combo_valid
        residual_crms = CRMS_Inband(y_valid - prediction_valid)
        
        if residual_crms < best_residual_crms:
            best_shift = shift
            best_gain = gain
            best_residual_crms = residual_crms
            best_prediction = gain * combo_shifted
            
    if best_shift is None:
        raise RuntimeError('Could not find a valid delayed CPS gain fit')
    return best_shift, best_gain, best_prediction, best_residual_crms

def _coherence(a, b, fs=FS):
    f, c = coherence(a, b, fs, nperseg=4096)
    return f, c