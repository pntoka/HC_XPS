from hc_xps.background import get_shirley_background, remove_background
from hc_xps.peak_fit import build_casa_lmfit_model, casa_fit_peaks, calculate_rsd
from hc_xps.plot import plot_basic_xps, plot_full_peak_fit, plot_xps_with_background


class XPSData:
    def __init__(self, energy, intensity, peak_config=None):
        '''
        Initialises the XPSData class with energy and intensity data.
        '''
        self.energy = energy
        self.intensity = intensity
        self.peak_config = peak_config
        self.background = None
        self.peak_fit_result = None
        self.energy_filtered = None
        self.intensity_filtered = None
        self.model_params = None
        self.model = None
        self.peaks_model = None

    def plot_raw_data(self):
        '''
        Plots the raw data.
        '''
        plot_basic_xps(self.energy, self.intensity)

    def get_background(self, start_energy, end_energy):
        '''
        Calculates the Shirley background and substracts it from data.
        '''
        self.background = get_shirley_background(self.energy, self.intensity, start_energy, end_energy)
        self.energy_filtered, self.intensity_filtered = remove_background(self.energy, self.intensity, start_energy, end_energy)
    
    def plot_data_with_background(self):
        '''
        Plots the data with the background.
        '''
        if self.background is None:
            raise ValueError("Background is not calculated. Run get_background() method first.")
        plot_xps_with_background(self.energy, self.intensity, self.background)

    def update_peak_config(self, peak_config):
        '''
        Updates the peak configuration.
        '''
        self.peak_config = peak_config

    def lmfit_model(self, peaks_model='6peaks_la', element='carbon', fixed_peaks=None, fixed_mix=False, mix=None):
        '''
        Builds the lmfit model.
        '''
        self.peaks_model = peaks_model
        fixed_peaks = ['C', 'D', 'E']
        self.model, self.model_params = build_casa_lmfit_model(self.peaks_model, element, fixed_peaks, fixed_mix, mix, self.peak_config)

    def fit_peaks(self, model='6peaks_la', method='least_squares', fixed_mix=False, mix=None):
        '''
        Fits the peaks.
        '''
        if self.energy_filtered is None:
            raise ValueError("Background is not calculated. Run get_background() method first.")
        self.lmfit_model(peaks_model=model, fixed_mix=fixed_mix, mix=mix)
        self.peak_fit_result = casa_fit_peaks(self.intensity_filtered, self.energy_filtered, self.model, self.model_params, method=method)
    
    def plot_peak_fit(self):
        '''
        Plots the peak fit.
        '''
        if self.peak_fit_result is None:
            raise ValueError("Peak fit is not calculated. Run fit_peaks() method first.")
        plot_full_peak_fit(self.peak_fit_result, self.energy, self.intensity, self.background, model=self.peaks_model, xps_config=self.peak_config)

    def rsd(self):
        '''
        Calculates the RSD.
        '''
        if self.peak_fit_result is None:
            raise ValueError("Peak fit is not calculated. Run fit_peaks() method first.")
        intensity_corrected = self.intensity_filtered - self.background[1]
        return calculate_rsd(intensity_corrected, self.peak_fit_result.best_fit)