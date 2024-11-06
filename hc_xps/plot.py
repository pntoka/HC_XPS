import matplotlib.pyplot as plt
import numpy as np
from hc_xps.peak_fit import get_peaks_config, calculate_rsd


def plot_basic_xps(energy, intensity):
    plt.plot(energy, intensity)
    plt.xlabel("Binding Energy (eV)")
    plt.ylabel("Intensity (a.u.)")
    plt.gca().invert_xaxis()
    plt.show()

def plot_xps_with_background(energy, intensity, background):
    plt.plot(energy, intensity, label='XPS Data')
    plt.plot(background[0], background[1], label='Background')
    plt.xlabel("Binding Energy (eV)")
    plt.ylabel("Intensity (a.u.)")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show()

def plot_full_peak_fit(result, energy, intensity, background, model='5peaks', element='carbon'):
    comps = result.eval_components(x=background[0])
    fig, (ax_residuals, ax_xps) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [1, 4]})
    peaks_config = get_peaks_config()
    peaks = peaks_config[element]['models'][model].split('+')
    for peak in peaks:
        ax_xps.plot(background[0], comps[f'{peak}_']+background[1], linestyle='--', label=peaks_config[element]['peaks'][peak]['docstring'].strip())
    ax_xps.plot(energy, intensity, label='XPS Data')
    ax_xps.plot(background[0], result.best_fit+background[1], label='Fit', linestyle='-.')
    ax_xps.plot(background[0], background[1], label='Background')
    ax_xps.legend()
    ax_xps.set_xlabel("Binding Energy (eV)")
    ax_xps.set_ylabel("Intensity (a.u.)")
    ax_xps.invert_xaxis()
    start_idx = np.argmin(abs(energy - background[0][0]))  # Getting the start and end positions for the intensity envelope
    end_idx = np.argmin(abs(energy - background[0][-1]))
    intensity_filtered = intensity[start_idx:end_idx+1]
    intensity_corrected = intensity_filtered - background[1]  # Removing the background from the intensity envelope
    rsd = calculate_rsd(intensity_corrected, result.best_fit)
    ax_residuals.plot(background[0], result.residual, color='black')
    ax_residuals.axhline(0, color='gray', linestyle='--')
    ax_residuals.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_residuals.get_yaxis().set_visible(False)
    ax_xps.spines['top'].set_visible(False)
    ax_residuals.spines['bottom'].set_visible(False)
    ax_residuals.text(0.9, 0.2, f'RSD: {rsd:.2f}', transform=ax_residuals.transAxes, fontsize=12, ha='center')
    ax_residuals.invert_xaxis()
    fig.subplots_adjust(hspace=0)
    plt.show()