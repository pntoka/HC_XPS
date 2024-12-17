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
    x_sorted = np.sort(x)
    la, lhm = LA(x_sorted, center, fwhm, alpha, beta, width=True)
    F_gaussian = lhm * ((1-mix)/mix)
    g = gaussian_norm(x_sorted, center, F_gaussian)
    la_conv = convolve(la, g, mode='full')
    la_conv_full = la_conv / np.max(la_conv)
    la_conv = np.interp(x_sorted, np.linspace(x.min(), x.max(), len(la_conv_full)), la_conv_full)
    unit_area = abs(np.trapezoid(la_conv, x_sorted))  # This is to get the unit area in order to then get peak height based on amplitude
    height = amplitude / unit_area if unit_area != 0 else 0
    if x[0] == x_sorted[0]:  # condition if x data is already ascending
        return la_conv * height
    return np.flip(la_conv * height)


def LA_peak(x, center, amplitude, fwhm, alpha, beta):
    w = 2 * fwhm / (np.sqrt(2 ** (1 / alpha) - 1) + np.sqrt(2 ** (1 / beta) - 1))
    la = generalized_lorentzian(x, center, w, alpha, beta)
    unit_area = abs(np.trapezoid(la, x))  # This is to get the unit area in order to then get peak height based on amplitude
    height = amplitude / unit_area if unit_area != 0 else 0
    return la * height


def calculate_imfp_tpp2m(kinetic_energy):
    """
    CODE FROM KherveFitting/libraries/Peak_Functions.py

    Function to  calculate IMFP using TPP-2M formula with average matrix parameters from XPS reference data.

    Parameters:
        kinetic_energy: electron energy in eV

    Returns:
        imfp: Inelastic Mean Free Path in nanometers

    Notes:
    Average matrix parameters derived from metals and inorganic compounds:
    - N_v = 4.684 (valence electrons per atom)
    - rho = 6.767 g/cm³ (density)
    - M = 137.51 g/mol (molecular weight)
    - E_g = 0 eV (bandgap energy)

    References:
    1. S.Tanuma, C.J.Powell and D.R.Penn, Surf. Interface Anal., 21, 165-176 (1993)
    2. Briggs & Grant, "Surface Analysis by XPS and AES" 2nd Ed., Wiley (2003), p.84-85
    """
    N_v = 4.684
    rho = 6.767
    M = 137.51
    E_g = 0

    E_p = 28.8 * np.sqrt((N_v * rho) / M)
    U = N_v * rho / M

    beta = -0.10 + 0.944 / (E_p ** 2 + E_g ** 2) ** 0.5 + 0.069 * rho ** 0.1
    gamma = 0.191 * rho ** (-0.5)
    C = 1.97 - 0.91 * U
    D = 53.4 - 20.8 * U

    imfp = kinetic_energy / (E_p ** 2 * (
            beta * np.log(gamma * kinetic_energy) -
            (C / kinetic_energy) +
            (D / kinetic_energy ** 2))) / 10  # Divide by 10 to convert from Å to nm
    return imfp


def calculate_normalised_area(binding_energy, area, rsf, txfn=1.0, photon_energy=1486.680054):
    '''
    Function to calculate the normalised area of a peak.
    '''
    kinetic_energy = photon_energy - binding_energy
    imfp = calculate_imfp_tpp2m(kinetic_energy)
    normalised_area = area * rsf * txfn * imfp
    return normalised_area

