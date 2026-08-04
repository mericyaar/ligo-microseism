import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import freqz

from ligo_microseism.config import FS, TAPS, FREQ_BAND, Causality_gap
from ligo_microseism.signal_utils import bandpass, CRMS_Inband

def FIR_Channel_Plot(w, X_std, y_std, channels, K=TAPS, gap=Causality_gap, fs=FS):
    C = len(channels)
    W = w.reshape(C, K)
    lags = np.arange(gap, K+gap) / fs

    rows = []

    for c, name in enumerate(channels):
        Dc_weights = W[c].sum() * (y_std/X_std[c])
        rows.append(dict(
            channel=name,
            FIR_DC_weights=Dc_weights
        ))

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4))
    for i in range(len(channels)):
        ax.plot(lags, W[i], lw=1.4, label=channels[i])
    ax.axhline(0, color='k', lw=0.7)
    ax.set_xlabel('Lags(s)')
    ax.set_ylabel('Weights')
    ax.legend()
    ax.grid(which='both', alpha=0.25)

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(rows)


def Blind_Plots(Results_library, cav, band=FREQ_BAND, time_chunk=(5000, 10000), gap = Causality_gap, K = TAPS, bandpass_plot=False):
    # Loop through the two blind data types
    for data_type in ['Blind_Noisy', 'Blind_Quiet']:

        # Skip if there are no folders loaded for this data type
        if data_type not in Results_library or not Results_library[data_type]:
            continue

        for name, folder_data in Results_library[data_type].items():
            f = folder_data['f']
            f_coh = folder_data['f_coh']

            type_label = data_type.replace('_', ' ')

            fig, ax = plt.subplots(3, 1, figsize=(10, 15))

            # First Plot: Physical ASD
            ax[0].loglog(f, folder_data['y_ASD'], color='navy', lw=2.4, zorder=6, label='Optical Target')
            ax[0].loglog(f, folder_data['yhat_ASD'], color='maroon', lw=1.8, zorder=4, ls='--', label='FIR')
            ax[0].loglog(f, folder_data['Res_ASD'], color='darkgreen', lw=1.6, zorder=3, ls=':', label='FIR Residual')
            if 'Delay_ASD' in folder_data:
                ax[0].loglog(f, folder_data['Delay_ASD'], color='darkorange', lw=1.8, zorder=4, ls='-.', label='Delay Predicted')
                ax[0].loglog(f, folder_data['Delay_Res_ASD'], color='gold', lw=1.6, zorder=3, ls=':', label='Delay Residual')
            
            ax[0].set_xlim(0.020, 0.5)
            ax[0].set_ylim(1e-9, 1e-4)
            ax[0].axvspan(band[0], band[1], color='gold', alpha=0.10, label=f'band {band[0]*1e3:.0f}-{band[1]*1e3:.0f} mHz')
            ax[0].set_xlabel('Frequency (Hz)')
            ax[0].set_ylabel(r'ASD (m/$\sqrt{\rm Hz}$)')
            ax[0].set_title(f'{cav} — blind folder {name} ({type_label}): ASD of Predicted, Optical and Residual')
            ax[0].grid(which='both', alpha=0.25)
            ax[0].legend(loc='lower left', fontsize=8)

            # Second plot: Time Series Chunk
            if bandpass_plot:
                yhat = bandpass(folder_data['yhat'])
            else:
                yhat = folder_data['yhat']
                
            time_ax = np.arange(time_chunk[0], time_chunk[1]) / FS
            ax[1].plot(time_ax, folder_data['y'][time_chunk[0]:time_chunk[1]] * 1e9, color='navy', zorder=6, label='Optical Target')
            ax[1].plot(time_ax, yhat[time_chunk[0]:time_chunk[1]] * 1e9, color='maroon', zorder=4, label='FIR')
            
            if 'Delay' in folder_data:
                ax[1].plot(time_ax, folder_data['Delay'][time_chunk[0]:time_chunk[1]] * 1e9, color='darkorange', lw=1.3, zorder=3, ls='-.', label='Delay')
            
            ax[1].set_xlabel('Time (s)')
            ax[1].set_ylabel('Amplitude (nm)')
            ax[1].set_title(f'{cav} — blind folder {name} ({type_label}): Time series of Optical vs Predicted')
            ax[1].grid(which='both', alpha=0.25)
            ax[1].legend(loc='upper right', fontsize=8)

            # Metrics Box
            metrics = (
                f'Mean 100–300 mHz BLRMS: {folder_data["mean_blrms"]:.0f} nm/s\n'
                f'CRMS Optical: {folder_data["CRMS_Real"]:.2e} m\n'
                f'CRMS FIR Residual: {folder_data["CRMS_Res"]:.2e} m\n'
                f'Suppression: {folder_data["supp"]: .2f} x\n'
                f'In-band Variance Explained: {folder_data["variance_explained_percent"]:.1f}%\n'
                f'Taps = {K}, Causality Gap = {gap}'
            )
            
            if 'CRMS_Delay_Res' in folder_data:
                metrics += (
                    f'\nCRMS Delay Res: {folder_data["CRMS_Delay_Res"]:.2e} m\n'
                    f'Delay Suppr : {folder_data["supp_Delay"]: .2f} x'
                )
                
            ax[1].text(0.05, 0.95, metrics, transform=ax[1].transAxes, fontsize=10,
                       ha='left', va='top', family='monospace', zorder=1000, 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=1.0))

            # Third Plot: Coherence
            ax[2].semilogx(f_coh, folder_data['FIR_coh'], color='maroon', lw=1.8, zorder=4, label='FIR Coherence')
            ax[2].semilogx(f_coh, folder_data['Combo_coh'], color='navy', lw=1.8, zorder=6, label='Combo Coherence')
            ax[2].set_xlabel('Frequency (Hz)')
            ax[2].set_ylabel('Coherence')
            ax[2].set_title(f'{cav} — blind folder {name} ({type_label}): Coherence of Predicted and Combo with Optical Target')
            ax[2].set_xlim(0.020, 0.5)
            ax[2].grid(which='both', alpha=0.25)
            ax[2].axvspan(band[0], band[1], color='gold', alpha=0.10, label=f'band {band[0]*1e3:.0f}-{band[1]*1e3:.0f} mHz')
            ax[2].legend(loc='upper right', fontsize=8)

            print(f"Mean 100–300 mHz BLRMS: {folder_data['mean_blrms']:.0f} nm/s")

            r = folder_data['y'] - folder_data['yhat']
            print('finite y:', np.isfinite(folder_data['y']).mean(),
                  ' finite yhat:', np.isfinite(folder_data['yhat']).mean(),
                  ' finite resid:', np.isfinite(r).mean())
            print('max |yhat|:', np.nanmax(np.abs(folder_data['yhat'])), ' max |y|:', np.nanmax(np.abs(folder_data['y'])))
            print('CRMS y:', CRMS_Inband(folder_data['y']), ' CRMS resid:', CRMS_Inband(r),
                  'supp:', CRMS_Inband(folder_data['y'])/CRMS_Inband(r))
            
            band_low_mhz = 1000.0 * band[0]
            band_high_mhz = 1000.0 * band[1]

            print(f"FIR explained {folder_data['variance_explained_percent']:.3f}% of the optical Variance in frequency band of {band_low_mhz:.0f}–{band_high_mhz:.0f} mHz: ")

            amp_ratio = np.nanmax(np.abs(folder_data['yhat'])) / np.nanmax(np.abs(folder_data['y']))
            if amp_ratio > 3:
                print(f' Prediction peak is {amp_ratio:.1f}× the true signal — out-of-band overfitting, ridge likely too low')

            plt.tight_layout()
            plt.show()


def Sweep_Plots(library, cav, channels, suppression_plot='Noisy', training_type='Noisy', band=FREQ_BAND, time_chunk=(5000, 10000)):
    # Extracts the nested dictionaries
    lofo_lib = library['LOFO_Training']
    other_lib = library.get('Other', {})

    lofo_names = list(lofo_lib.keys())
    other_names = list(other_lib.keys())

    # Calculates Vector Math only on the LOFO training folds
    W = np.array([lofo_lib[name]['weights'] for name in lofo_names])
    w_mean = np.mean(W, axis=0)
    C = len(channels)
    K = TAPS

    W_reshaped = W.reshape(len(lofo_names), C, K)
    w_mean_reshaped = w_mean.reshape(C, K)

    # Plot 1: Bode Plot
    fig, ax = plt.subplots(2, C, figsize=(4.5 * C, 6.5), sharex=True)

    # Safeguard for single-channel cavities
    if C == 1:
        ax = ax[:, np.newaxis]

    cmap = plt.cm.plasma(np.linspace(0, 0.9, len(lofo_names)))

    for c in range(C):
        ax_mag = ax[0, c]
        ax_phase = ax[1, c]

        # Plots individual LOFO splits
        for k in range(len(lofo_names)):
            freqs, H = freqz(W_reshaped[k, c, :], a=1, fs=FS, worN=1024)
            mag = 20 * np.log10(np.abs(H) + 1e-10)
            phase = np.unwrap(np.angle(H)) * (180 / np.pi)

            ax_mag.semilogx(freqs, mag, color=cmap[k], lw=0.9, alpha=0.5)
            ax_phase.semilogx(freqs, phase, color=cmap[k], lw=0.9, alpha=0.5)

        # Plots the Mean Filter over the top
        freqs, H_mean = freqz(w_mean_reshaped[c, :], a=1, fs=FS, worN=1024)
        ax_mag.semilogx(freqs, 20 * np.log10(np.abs(H_mean) + 1e-10), 'k-', lw=2.2, label='Mean Filter')
        ax_phase.semilogx(freqs, np.unwrap(np.angle(H_mean)) * (180 / np.pi), 'k-', lw=2.2)

        ax_mag.set_title(f"Channel: {channels[c]}", fontweight='bold')
        ax_mag.axvspan(band[0], band[1], color='gold', alpha=0.15)
        ax_phase.axvspan(band[0], band[1], color='gold', alpha=0.15)
        ax_mag.grid(which='both', alpha=0.25)
        ax_phase.grid(which='both', alpha=0.25)

        if c == 0:
            ax_mag.set_ylabel('Magnitude (dB)')
            ax_phase.set_ylabel('Phase (deg)')
            ax_mag.legend(loc='lower left', fontsize=8)

        ax_phase.set_xlabel("Frequency (Hz)")
        ax_phase.set_xlim(0.01, 0.5)

    fig.suptitle(f"{cav} FIR Frequency Response Stability Across {len(lofo_names)} Folds", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Plot 2: Suppression
    plot_selection = suppression_plot.strip().capitalize()
    valid_plot_selections = {'Noisy', 'Quiet', 'All'}

    if plot_selection not in valid_plot_selections:
        raise ValueError("suppression_plot must be 'Noisy', 'Quiet', or 'All'")

    training_data_type = training_type.strip().capitalize()
    other_data_type = 'Quiet' if training_data_type == 'Noisy' else 'Noisy'

    plot_training = plot_selection in {training_data_type, 'All'}
    plot_other = plot_selection in {other_data_type, 'All'}

    if plot_other and not other_names:
        raise ValueError(
            f"suppression_plot='{plot_selection}' requires the "
            f"{other_data_type} Sweep predictions, but they were not "
            "evaluated. Set eval_other = True."
        )

    fig, ax = plt.subplots(figsize=(7, 4))
    plotted_suppression = []
    plotted_any = False

    if plot_training:
        lofo_supp = np.array([lofo_lib[name]['supp'] for name in lofo_names], dtype=float)
        ax.scatter(lofo_names, lofo_supp, color='navy', s=50, marker='o', zorder=5, label=f'{training_data_type} LOFO predictions')
        plotted_suppression.append(lofo_supp)
        plotted_any = True

    if plot_other:
        other_supp = np.array([other_lib[name]['supp'] for name in other_names], dtype=float)

        if plot_training:
            ax.axvline(x=len(lofo_names) - 0.5, color='gray', linestyle='--', lw=1.2, alpha=0.6)

        ax.scatter(other_names, other_supp, color='maroon', s=100, marker='*', zorder=5, label=f'{other_data_type} master-filter predictions')
        plotted_suppression.append(other_supp)
        plotted_any = True

    if not plotted_any:
        raise ValueError('No suppression predictions were selected for plotting')

    all_plotted_suppression = np.concatenate(plotted_suppression)
    finite_suppression = all_plotted_suppression[np.isfinite(all_plotted_suppression)]

    if finite_suppression.size == 0:
        raise ValueError('No finite suppression values are available to plot')

    # Mean suppression and standard error of the mean
    mean_suppression = np.mean(finite_suppression)

    if finite_suppression.size > 1:
        suppression_standard_error = np.std(finite_suppression, ddof=1) / np.sqrt(finite_suppression.size)
    else:
        suppression_standard_error = np.nan

    print("\n" + "=" * 64)
    print(f"{cav} SWEEP SUPPRESSION STATISTICS")
    print("=" * 64)
    print(f"Number of plotted blind realizations : {finite_suppression.size}")
    print(f"Mean suppression                    : {mean_suppression:.2f} ± {suppression_standard_error:.2f}")
    print("=" * 64)

    suppression_ylim_upper = np.max(finite_suppression)*1.25

    ax.grid(which='both', alpha=0.25)
    ax.set_title(f'{cav} Suppression: Train {training_data_type}, Plot {plot_selection}', fontweight='bold')
    ax.set_xlabel('GPS Dates')
    ax.set_ylabel('In-Band Suppression (x)')
    ax.set_ylim(0.0, suppression_ylim_upper)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Plot 3: Suppression versus mean 100–300 mHz BLRMS
    fig, ax = plt.subplots(figsize=(6, 5))
    plotted_blrms = False

    if plot_training:
        lofo_mean_blrms = np.array([lofo_lib[name]['mean_blrms'] for name in lofo_names], dtype=float)
        lofo_suppression = np.array([lofo_lib[name]['supp'] for name in lofo_names], dtype=float)

        lofo_mask = (np.isfinite(lofo_mean_blrms) & np.isfinite(lofo_suppression))

        if np.any(lofo_mask):
            ax.scatter(lofo_mean_blrms[lofo_mask], lofo_suppression[lofo_mask], color='navy', s=50, marker='o', zorder=5, label=f'{training_data_type} LOFO predictions')
            plotted_blrms = True

    if plot_other:
        other_mean_blrms = np.array([other_lib[name]['mean_blrms'] for name in other_names], dtype=float)
        other_suppression = np.array([other_lib[name]['supp'] for name in other_names], dtype=float)

        other_mask = (np.isfinite(other_mean_blrms) & np.isfinite(other_suppression))

        if np.any(other_mask):
            ax.scatter(other_mean_blrms[other_mask], other_suppression[other_mask], color='maroon', s=50, marker='*', zorder=5, label=f'{other_data_type} master-filter predictions')
            plotted_blrms = True

    if not plotted_blrms:
        raise ValueError('No finite BLRMS and suppression pairs are available to plot')

    ax.grid(which='both', alpha=0.25)
    ax.set_title(f'{cav}: Suppression vs Mean 100–300 mHz BLRMS', fontweight='bold')
    ax.set_xlabel('Mean 100–300 mHz BLRMS')
    ax.set_ylabel('In-Band Suppression (x)')
    ax.legend(loc='best', framealpha=0.9)
    ax.set_ylim(0.0, suppression_ylim_upper)

    plt.tight_layout()
    plt.show()