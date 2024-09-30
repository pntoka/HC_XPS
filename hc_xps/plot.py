import matplotlib.pyplot as plt
import numpy as np
from peak_fit import get_peaks_config


def plot_basic_xps(energy, intensity):
    plt.plot(energy, intensity)
    plt.xlabel("Binding Energy (eV)")
    plt.ylabel("Intensity (a.u.)")
    plt.gca().invert_xaxis()
    plt.show()

def plot_xps_with_background(energy, intensity, background):
    plt.plot(energy, intensity, label='XPS Data')
    plt.plot(energy, background, label='Background')
    plt.xlabel("Binding Energy (eV)")
    plt.ylabel("Intensity (a.u.)")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show()

def plot_full_peak_fit(result, energy, intensity, background, model='5peaks', element='carbon'):
    comps = result.eval_components(x=energy)
    fig, ax = plt.subplots()
    peaks_config = get_peaks_config()
    peaks = peaks_config[element]['models'][model].split('+')
    for peak in peaks:
        ax.plot(energy, comps[f'{peak}_']+background, linestyle='--', label=peaks_config[element]['peaks'][peak]['docstring'])
    ax.plot(energy, result.best_fit+background, label='Fit', linestyle='-.')
    ax.plot(energy, intensity, label='XPS Data')
    ax.plot(energy, background, label='Background')
    ax.legend()
    ax.set_xlabel("Binding Energy (eV)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.invert_xaxis()
    plt.show()