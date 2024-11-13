import numpy as np
from scipy.signal import convolve


def lorentzian_norm(x, center, F):
    """
    Lorentzian function normalised to peak height 1
    """
    result = 1 / (1 + 4 * ((x - center)/F)**2)
    return result


def gaussian_norm(x, center, F):
    """
    Gaussian function normalised to peak height 1
    """
    result = np.exp(-4 * np.log(2) * ((x - center)**2/F**2))
    return result


def generalized_lorentzian(x, center, F, alpha, beta):
    """
    Generalized Lorentzian function where each side is raised to a different power
    """
    l_x = lorentzian_norm(x, center, F)
    result = np.where(x <= center, l_x**alpha, l_x**beta)
    return result


def LA(x, center, fwhm, alpha, beta, width=False):
    w = 2 * fwhm / (np.sqrt(2 ** (1 / alpha) - 1) + np.sqrt(2 ** (1 / beta) - 1))
    la = generalized_lorentzian(x, center, w, alpha, beta)
    if width:
        return la, w
    return la


def LA_conv(x, center, amplitude, fwhm, alpha, beta, mix):
    la, lhm = LA(x, center, fwhm, alpha, beta, width=True)
    F_gaussian = lhm * ((1-mix)/mix)
    g = gaussian_norm(x, center, F_gaussian)
    la_conv = convolve(la, g, mode='same')
    la_conv = la_conv / np.max(la_conv)
    unit_area = abs(np.trapz(la_conv, x))  # This is to get the unit area in order to then get peak height based on amplitude
    height = amplitude / unit_area if unit_area != 0 else 0
    return la_conv * height