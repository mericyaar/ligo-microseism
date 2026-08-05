import numpy as np
import matplotlib.pyplot as plt

from ligo_microseism.config import FS, FREQ_BAND, TAPS, Alphas, Run_Delay, Causality_gap
from ligo_microseism.signal_utils import compute_ASD, CRMS_Inband, _coherence, best_delay_and_gain
from ligo_microseism.fir import FIR_Prediction, FIR_Normal_Eq, _safe_solve

def LOFO(library, alphas=Alphas, band=FREQ_BAND):
    Train_names = list(library['Training'].keys())

    X_all = np.concatenate([library['Training'][name]['X'] for name in Train_names], axis=0)
    y_all = np.concatenate([library['Training'][name]['y'] for name in Train_names], axis=0)

    X_Mean = X_all.mean(axis=0)
    X_Std = X_all.std(axis=0)
    y_Mean = y_all.mean(axis=0)
    y_Std = y_all.std(axis=0)

    X_Z = lambda X: (X - X_Mean) / X_Std
    y_Z = lambda y: (y - y_Mean) / y_Std

    X_Train_Z = {f: X_Z(library['Training'][f]['X']) for f in Train_names}
    y_Train_Z = {f: y_Z(library['Training'][f]['y']) for f in Train_names}

    Eq_Dict = {f: FIR_Normal_Eq(X_Train_Z[f], y_Train_Z[f]) for f in Train_names}

    A_Tot = np.zeros_like(Eq_Dict[Train_names[0]][0])
    b_Tot = np.zeros_like(Eq_Dict[Train_names[0]][1])
    n_Tot = 0

    for f in Train_names:
        A_Tot += Eq_Dict[f][0]
        b_Tot += Eq_Dict[f][1]
        n_Tot += Eq_Dict[f][2]

    LOFO_Base = {
        f: (A_Tot - Eq_Dict[f][0], b_Tot - Eq_Dict[f][1], n_Tot - Eq_Dict[f][2])
        for f in Train_names
    }

    D = A_Tot.shape[0]
    ridge_cand = sorted(set(alphas), reverse=True)
    last_good_alpha = None
    last_good_score = None
    tested_alphas = []
    score_list = []
    
    for alpha in ridge_cand:
        score = 0.0
        for f in Train_names:
            M_base, rhs_lofo, n_lofo = LOFO_Base[f]
            M_lofo = M_base.copy()
            M_lofo.flat[::D + 1] += alpha * n_lofo
            w_lofo = _safe_solve(M_lofo, rhs_lofo)

            yhat = FIR_Prediction(X_Train_Z[f], w_lofo)
            mask = ~np.isnan(yhat)
            score += CRMS_Inband(y_Train_Z[f][mask] - yhat[mask])

        score /= len(Train_names)
        tested_alphas.append(alpha)
        score_list.append(score)
        last_good_alpha = alpha
        last_good_score = score
        print(f' Alpha {alpha:.0e} -> Avg LOFO CRMS: {score:.4f}')

    if last_good_alpha is None:
        raise ValueError("Model instability in LOFO sweep.")


    final_alpha = last_good_alpha
    print(f'\nWinning Alpha: {final_alpha:.0e} | Avg LOFO CRMS: {last_good_score:.4f}')

    M_final = A_Tot.copy()
    M_final.flat[::D + 1] += final_alpha * n_Tot
    final_w = _safe_solve(M_final, b_Tot)

    return final_w, final_alpha, X_Mean, X_Std, y_Mean, y_Std


def Blind_Eval(library, w, alpha, X_Mean, X_Std, y_Mean, y_Std, run_delay=Run_Delay):
    Results_library = {
        'Blind_Noisy': {},
        'Blind_Quiet': {}
    }
    
    for regime in ['Blind_Noisy', 'Blind_Quiet']:
        if regime not in library or not library[regime]:
            continue

        for name, data in library[regime].items():
            print(f'Evaluating {name}...')
            X_Blind = (data['X'] - X_Mean) / X_Std
            y_Blind = (data['y'] - y_Mean) / y_Std

            yhat = FIR_Prediction(X_Blind, w)
            yhat_real = yhat * y_Std + y_Mean
            y_real = data['y']

            mask = ~np.isnan(yhat)
            yhat_real = yhat_real[mask]
            y_real = y_real[mask]
            combo = data['combo'][mask]

            f, yhat_real_ASD = compute_ASD(yhat_real, fs=FS)
            f, y_real_ASD = compute_ASD(y_real, fs=FS)
            f, Res_ASD = compute_ASD(yhat_real - y_real, fs=FS)

            CRMS_Real = CRMS_Inband(y_real)
            CRMS_Res = CRMS_Inband(yhat_real - y_real)
            supp = CRMS_Real / CRMS_Res

            variance_explained = 1.0 - (CRMS_Res/CRMS_Real)**2
            variance_explained_percent = 100.0 * variance_explained

            f_coh, FIR_coh = _coherence(y_real, yhat_real)
            f_coh, Combo_coh = _coherence(y_real, combo)

            Results_library[regime][name] = dict(
                yhat=yhat_real, y=y_real, f=f, yhat_ASD=yhat_real_ASD,
                y_ASD=y_real_ASD, Res_ASD=Res_ASD, CRMS_Real=CRMS_Real,
                CRMS_Res=CRMS_Res, mean_blrms=data['mean_blrms'], supp=supp,
                variance_explained=variance_explained,
                variance_explained_percent=variance_explained_percent,
                f_coh=f_coh, FIR_coh=FIR_coh, Combo_coh=Combo_coh
            )

            if run_delay:
                shift, g, Delay, _ = best_delay_and_gain(combo, y_real)
                _, Delay_ASD = compute_ASD(Delay, fs=FS)
                CRMS_Delay_Res = CRMS_Inband(Delay - y_real)
                _, Delay_Res_ASD = compute_ASD(Delay - y_real, fs=FS)

                Results_library[regime][name].update({
                    'Delay': Delay, 'Delay_ASD': Delay_ASD,
                    'Delay_Res_ASD': Delay_Res_ASD, 'CRMS_Delay_Res': CRMS_Delay_Res,
                    'supp_Delay': CRMS_Real / CRMS_Delay_Res
                })

    return Results_library


def Sweep_Cavity(library, alphas=Alphas, band=FREQ_BAND, eval_other=False, run_delay=Run_Delay):

    print("Folder evaluation is in progress...")
    final_w, final_alpha, X_Mean, X_Std, y_Mean, y_Std = LOFO(library, alphas=alphas, band=band)

    
    Train_names = list(library['Training'].keys())
    X_Z = lambda X: (X - X_Mean) / X_Std
    
    Eq_Dict = {f: FIR_Normal_Eq(X_Z(library['Training'][f]['X']), (library['Training'][f]['y'] - y_Mean) / y_Std) for f in Train_names}

    A_Tot = np.zeros_like(Eq_Dict[Train_names[0]][0])
    b_Tot = np.zeros_like(Eq_Dict[Train_names[0]][1])
    n_Tot = 0

    for f in Train_names:
        A_Tot += Eq_Dict[f][0]
        b_Tot += Eq_Dict[f][1]
        n_Tot += Eq_Dict[f][2]

    LOFO_Base = {
        f: (A_Tot - Eq_Dict[f][0], b_Tot - Eq_Dict[f][1], n_Tot - Eq_Dict[f][2])
        for f in Train_names
    }
    
    D = A_Tot.shape[0]

    def evaluate_folder(w_vector, raw_X, raw_y, raw_combo, mean_blrms):
        X_Blind = (raw_X - X_Mean) / X_Std
        yhat = FIR_Prediction(X_Blind, w_vector)
        yhat_real = yhat * y_Std + y_Mean

        mask = ~np.isnan(yhat_real)
        yhat_real = yhat_real[mask]
        y_real = raw_y[mask]
        combo_masked = raw_combo[mask]

        f, yhat_real_ASD = compute_ASD(yhat_real, fs=FS)
        f, y_real_ASD = compute_ASD(y_real, fs=FS)
        f, Res_ASD = compute_ASD(yhat_real - y_real, fs=FS)

        CRMS_Real = CRMS_Inband(y_real)
        CRMS_Res = CRMS_Inband(yhat_real - y_real)
        supp = CRMS_Real / CRMS_Res
        variance_explained = 1.0 - (CRMS_Res/CRMS_Real)**2
        variance_explained_percent = 100.0 * variance_explained

        res = dict(
            weights=w_vector, yhat=yhat_real, y=y_real, f=f,
            yhat_ASD=yhat_real_ASD, y_ASD=y_real_ASD, Res_ASD=Res_ASD,
            CRMS_Real=CRMS_Real, CRMS_Res=CRMS_Res, supp=supp,
            variance_explained=variance_explained,
            variance_explained_percent=variance_explained_percent,
            mean_blrms=mean_blrms,
        )

        if run_delay:
            shift, g, Delay, _ = best_delay_and_gain(combo_masked, y_real)
            _, Delay_ASD = compute_ASD(Delay, fs=FS)
            CRMS_Delay_Res = CRMS_Inband(Delay - y_real)
            _, Delay_Res_ASD = compute_ASD(Delay - y_real, fs=FS)

            res.update({
                'Delay': Delay, 'Delay_ASD': Delay_ASD, 
                'Delay_Res_ASD': Delay_Res_ASD, 'CRMS_Delay_Res': CRMS_Delay_Res, 
                'supp_Delay': CRMS_Real / CRMS_Delay_Res
            })

        return res

    Sweep_Library = {'LOFO_Training': {}, 'Other': {}}

    print("Evaluating LOFO Training splits")
    for name in Train_names:
        M_base, rhs_lofo, n_lofo = LOFO_Base[name]
        M_lofo_final = M_base.copy()
        M_lofo_final.flat[::D + 1] += final_alpha * n_lofo
        w = _safe_solve(M_lofo_final, rhs_lofo)
        data = library['Training'][name]
        Sweep_Library['LOFO_Training'][name] = evaluate_folder(
            w, data['X'], data['y'], data['combo'], data['mean_blrms']
        )

    if eval_other:
        print("Evaluating Master Filter on Other sets")
        M_master = A_Tot.copy()
        M_master.flat[::D + 1] += final_alpha * n_Tot
        w_master = _safe_solve(M_master, b_Tot)

        if 'Other' in library and library['Other']:
            for name, data in library['Other'].items():
                Sweep_Library['Other'][name] = evaluate_folder(
                    w_master, data['X'], data['y'], data['combo'], data['mean_blrms']
                )
                Sweep_Library['Other'][name]['data_type'] = 'Other'

    print('Fitting is complete.')
    return Sweep_Library