from pathlib import Path
from scipy.signal import butter

# :::::::::::::: Parameters :::::::::::::::::::
FS = 4.0 #Hz
DURATION = 14400 #seconds
FILTER_ORDER = 2
FILTER_HP_FREQ = 0.015 # Hz freq of Highpass
FILTER_BAND = (0.025, 0.270) #Hz Freq of Bandpass
FREQ_BAND = (0.030, 0.250) # Where the suppression and CRMS are evaluated
TRIM_Seconds = 400 #seconds how much to trim from each ends

#:::::::::: Controls ::::::::::::::::::
Cavity = 'DARM'
Run_Type = 'Blind'
Training_Type = 'Noisy'


# Whether to bandpass the folders, if it is false then it highpasses
Bandpass_Training = True
# !!! Bandpass Blind does not apply to Sweep !!!
Bandpass_Blind = True

Run_Delay = False
# :::::::::::: Blind Indices :::::::::::
Noisy_Blind_idx = [2,6]
Quiet_Blind_idx = []
Causality_gap = 1 # how many samples gap should the input and output have (in samples i.e. 1=0.25s for 4 Hz) 
# ----FIR----
Lookback = 66 #seconds The past FIR engine has access to
Alphas = (1e-7, 1e-5) # Ridge strength list

# :::::::::::::: Converted Parameters :::::::::::::::::::
TRIM = int(TRIM_Seconds*FS)
TAPS = int(Lookback*FS) + 1

# :::::::::::::: Filters ::::::::::::::::
_HP_SOS  = butter(FILTER_ORDER, FILTER_HP_FREQ, btype='highpass', fs=FS, output='sos')
_LOSS_BP_SOS = butter(FILTER_ORDER, list(FILTER_BAND), btype='bandpass', fs=FS, output='sos')

# :::::::::::::: Definitions ::::::::::::::::
Base_Dir = Path('/Users/mericyasar/ligo-microseism')

# :::::::::::: Data Directories :::::::::::
Noisy_Data = ['1424916018',
              '1420675218',
              '1422266402',
              '1417615060',
              '1417838418',
              '1422316818',
              '1424689218',
              '1420416018',
              '1418934929',
              '1420851618',
              '1424736018',
              '1417846196',
              '1420344018',
              '1423267218',
              '1423274418',
             ]

Quiet_Data = [
    '1435708818', '1435795218', '1436227218', '1440460818', '1433462418',
    '1434067218', '1435276818', '1435903098', '1435968018', '1436400018',
    '1437652918', '1434426707', '1434512334', '1434679218', '1434844818'
]