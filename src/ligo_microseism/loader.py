import numpy as np
from pathlib import Path

from ligo_microseism.config import (
    Base_Dir, Quiet_Data, Noisy_Data, 
    Bandpass_Training, Bandpass_Blind,
    Noisy_Blind_idx, Quiet_Blind_idx
)
from ligo_microseism.signal_utils import load_sensor, highpass, bandpass

def load_cavity(date_folder, data_type, filter_type, cav, base_dir=Base_Dir):
    if data_type == 'Noisy':
        p = Path(base_dir) / "data/Noisy" / date_folder
    elif data_type == 'Quiet':
        p = Path(base_dir) / "data/Quiet" / date_folder
    else:
        raise ValueError('Invalid run type')

    # Load data
    opt_sensors = {}
    cps_sensors = {}
    sensor_freq = {}

    for a in cav['OPT_Channels']:
        raw_data, org_fs = load_sensor(p, a + '.txt')
        opt_sensors[a] = raw_data
        sensor_freq[a] = org_fs

    for b in cav['CPS_Channels']:
        raw_data, org_fs = load_sensor(p, b + '.txt')
        cps_sensors[b] = raw_data * 1e-9
        sensor_freq[b] = org_fs

    raw_blrms, _ = load_sensor(p, 'CS100M300MBLRMS.txt')
    mean_blrms = np.mean(raw_blrms)

    if len(set(sensor_freq.values())) > 1:
        raise ValueError('Frequency mismatch, the group delay caused by decimation will be different!')

    arr_list = list(opt_sensors.values()) + list(cps_sensors.values())
    n = min([len(d) for d in arr_list])
    opt_sensors = {a: opt_sensors[a][:n] for a in cav['OPT_Channels']}
    cps_sensors = {b: cps_sensors[b][:n] for b in cav['CPS_Channels']}

    for nm, arr in list(opt_sensors.items()) + list(cps_sensors.items()):
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = 0.0
            print(f'There were nan/inf in {nm} - amount = {bad.sum()}, got changed to 0.0')

    if filter_type == 'Bandpass':
        opt_sensors = {a: bandpass(opt_sensors[a]) for a in cav['OPT_Channels']}
        cps_sensors = {b: bandpass(cps_sensors[b]) for b in cav['CPS_Channels']}
    elif filter_type == 'Highpass':
        opt_sensors = {a: highpass(opt_sensors[a]) for a in cav['OPT_Channels']}
        cps_sensors = {b: highpass(cps_sensors[b]) for b in cav['CPS_Channels']}
    else:
        raise ValueError('Invalid filter type')

    X = np.column_stack([cps_sensors[cnm] for cnm in cav['CPS_Channels']])
    y = cav['target_func'](opt_sensors)
    combo = cav['combo_func'](cps_sensors)

    if y.std() == 0:
        raise ValueError('Dead Channel')
    for nm, c in cps_sensors.items():
        if c.std() == 0:
            raise ValueError(f'Dead Channel {nm}')
            
    print(f"Loaded {date_folder}")
    return date_folder, X, y, combo, mean_blrms


def build_library(cav, Run_Type, Training_Type, base_dir=Base_Dir, quiet_data_list=Quiet_Data, 
                  noisy_data_list=Noisy_Data, quiet_blind_idx=Quiet_Blind_idx, noisy_blind_idx=Noisy_Blind_idx, Eval_Other=False):

    Data_Library = {
        'Training': {},
        'Blind_Quiet': {},
        'Blind_Noisy': {}
    }

    Noisy_Blind_Data = [d for i, d in enumerate(noisy_data_list) if i in noisy_blind_idx]
    Quiet_Blind_Data = [d for i, d in enumerate(quiet_data_list) if i in quiet_blind_idx]

    if Training_Type == 'Noisy':
        if Run_Type == 'Blind':
            Training_Data = [d for i, d in enumerate(noisy_data_list) if i not in noisy_blind_idx]
        elif Run_Type == 'Sweep':
            Training_Data = noisy_data_list
            Other_Data = quiet_data_list if Eval_Other else []

    elif Training_Type == 'Quiet':
        if Run_Type == 'Blind':
            Training_Data = [d for i, d in enumerate(quiet_data_list) if i not in quiet_blind_idx]
        elif Run_Type == 'Sweep':
            Training_Data = quiet_data_list
            Other_Data = noisy_data_list if Eval_Other else []
    else:
        raise ValueError('Invalid Training_Type')

    # Load Training Pool
    for d in Training_Data:
        try:
            _, X, y, combo, mean_blrms = load_cavity(d, data_type=Training_Type, filter_type='Bandpass' if Bandpass_Training else 'Highpass', cav=cav)
            Data_Library['Training'][d] = {'X': X, 'y': y, 'combo': combo, 'mean_blrms': mean_blrms}
        except Exception as e:
            print(f"Error loading Training folder {d}: {e}")

    print(f"Loaded {len(Data_Library['Training'])} {Training_Type} folders for fitting")

    # Load Blind Pools
    if Run_Type == 'Blind':
        for d in Noisy_Blind_Data:
            if d in Data_Library['Training']:
                continue
            try:
                _, X, y, combo, mean_blrms = load_cavity(d, data_type='Noisy', filter_type='Bandpass' if Bandpass_Blind else 'Highpass', cav=cav)
                Data_Library['Blind_Noisy'][d] = {'X': X, 'y': y, 'combo': combo, 'mean_blrms': mean_blrms}
            except Exception as e:
                print(f"Error loading Blind (Noisy) folder {d}: {e}")

        for d in Quiet_Blind_Data:
            if d in Data_Library['Training']:
                continue
            try:
                _, X, y, combo, mean_blrms = load_cavity(d, data_type='Quiet', filter_type='Bandpass' if Bandpass_Blind else 'Highpass', cav=cav)
                Data_Library['Blind_Quiet'][d] = {'X': X, 'y': y, 'combo': combo, 'mean_blrms': mean_blrms}
            except Exception as e:
                print(f"Error loading Blind (Quiet) folder {d}: {e}")

    elif Run_Type == 'Sweep':
        Sweep_Other = 'Blind_Noisy' if Training_Type == 'Quiet' else 'Blind_Quiet'
        for d in Other_Data:
            try:
                _, X, y, combo, mean_blrms = load_cavity(d, data_type='Noisy' if Training_Type == 'Quiet' else 'Quiet', filter_type='Bandpass' if Bandpass_Blind else 'Highpass', cav=cav)
                Data_Library[Sweep_Other][d] = {'X': X, 'y': y, 'combo': combo, 'mean_blrms': mean_blrms}
            except Exception as e:
                print(f"Error loading Blind folder {d}: {e}")

    print(f"\nLoaded total of {len(Data_Library['Training']) + len(Data_Library['Blind_Noisy']) + len(Data_Library['Blind_Quiet'])} folders into library")
    return Data_Library