''''
This module contains the functions to perform background subtraction on XPS data.
Function to calculate the shirley background is adapted from https://github.com/chstan/arpes/blob/master/arpes/analysis/shirley.py
'''
import numpy as np
import warnings


def _calculate_shirley_background_full_range(
    xps: np.ndarray, eps=1e-7, max_iters=50, n_samples=5
) -> np.ndarray:
    """Core routine for calculating a Shirley background on np.ndarray data."""
    background = np.copy(xps)
    cumulative_xps = np.cumsum(xps, axis=0)
    total_xps = np.sum(xps, axis=0)

    rel_error = np.inf

    i_left = np.mean(xps[:n_samples], axis=0)
    i_right = np.mean(xps[-n_samples:], axis=0)

    iter_count = 0

    k = i_left - i_right
    for iter_count in range(max_iters):
        cumulative_background = np.cumsum(background, axis=0)
        total_background = np.sum(background, axis=0)

        new_bkg = np.copy(background)

        for i in range(len(new_bkg)):
            new_bkg[i] = i_right + k * (
                (total_xps - cumulative_xps[i] - (total_background - cumulative_background[i]))
                / (total_xps - total_background + 1e-5)
            )

        rel_error = np.abs(np.sum(new_bkg, axis=0) - total_background) / (total_background)

        background = new_bkg

        if np.any(rel_error < eps):
            break

    if (iter_count + 1) == max_iters:
        warnings.warn(
            "Shirley background calculation did not converge "
            + "after {} steps with relative error {}!".format(max_iters, rel_error)
        )

    return background


def _find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def prepare_xps(start_energy, end_energy, energy, intensity):
    """
    Function to take xps data and make end points equal to the average of the 2 points before and after the end points.
    This is equivalent of taking the average of 0.5 eV of points arounf the end points.
    """
    start_idx = _find_nearest(energy, start_energy)
    end_idx = _find_nearest(energy, end_energy)
    # start_idx = np.where(energy == start_energy)[0][0]
    # end_idx = np.where(energy == end_energy)[0][0]
    i_left = np.mean(intensity[start_idx-2:start_idx+2])
    i_right = np.mean(intensity[end_idx-2:end_idx+2])
    new_xps = np.copy(intensity[start_idx:end_idx])
    new_xps[0] = i_left
    new_xps[-1] = i_right
    return new_xps, start_idx, end_idx


def get_shirley_background(energy, intensity, start_energy, end_energy):
    """
    Function to calculate shirley background of xps data.
    """
    xps, start_idx, end_idx = prepare_xps(start_energy, end_energy, energy, intensity)
    background = _calculate_shirley_background_full_range(xps)
    energy_bkg = energy[start_idx:end_idx]
    bkg = np.vstack((energy_bkg, background))
    return bkg


def remove_background(energy, intensity, start_energy, end_energy):
    """
    Function to remove shirley background from xps data.
    """
    xps, start_idx, end_idx = prepare_xps(start_energy, end_energy, energy, intensity)
    background = _calculate_shirley_background_full_range(xps)
    intensity = intensity[start_idx:end_idx]
    intensity = intensity - background
    return energy[start_idx:end_idx], intensity