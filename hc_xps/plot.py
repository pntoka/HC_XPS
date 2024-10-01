import matplotlib.pyplot as plt
import numpy as np
from hc_xps.peak_fit import get_peaks_config


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
    fig, ax = plt.subplots()
    peaks_config = get_peaks_config()
    peaks = peaks_config[element]['models'][model].split('+')
    for peak in peaks:
        ax.plot(background[0], comps[f'{peak}_']+background[1], linestyle='--', label=peaks_config[element]['peaks'][peak]['docstring'].strip())
    ax.plot(background[0], result.best_fit+background[1], label='Fit', linestyle='-.')
    ax.plot(energy, intensity, label='XPS Data')
    ax.plot(background[0], background[1], label='Background')
    ax.legend()
    ax.set_xlabel("Binding Energy (eV)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.invert_xaxis()
    plt.show()