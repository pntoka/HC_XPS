from hc_xps.background import get_shirley_background, remove_background
from hc_xps.peak_fit import build_casa_lmfit_model, casa_fit_peaks, calculate_rsd, find_best_oxygen_fit, extract_fit_results, build_oxygen_lmfit_model, fit_oxygen_peaks
from hc_xps.plot import plot_basic_xps, plot_full_peak_fit, plot_xps_with_background, full_peak_with_table_fig
from hc_xps.utils import extract_energy_intensity, calculate_ratios
import pickle


class XPSData:
    def __init__(self, element, energy, intensity, peak_config=None):
        '''
        Initialises the XPSData class with energy and intensity data.
        '''
        self.element = element
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
        self.peak_table = None

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

    def lmfit_model(self, peaks_model='6peaks_la', fixed_peaks=None, fixed_mix=False, mix=None):
        '''
        Builds the lmfit model.
        '''
        element = self.element
        if element == 'carbon':
            self.peaks_model = peaks_model
            fixed_peaks = ['C', 'D', 'E']
            self.model, self.model_params = build_casa_lmfit_model(self.peaks_model, element, fixed_peaks, fixed_mix, mix, self.peak_config)
        elif element == 'oxygen':
            if peaks_model == '2peaks' or peaks_model == '3peaks' or peaks_model == '4peaks':
                self.peaks_model = peaks_model
                self.model, self.model_params = build_oxygen_lmfit_model(self.peaks_model, element, self.peak_config)

    def fit_peaks(self, model='6peaks_la', method='least_squares', fixed_mix=False, mix=None):
        '''
        Fits the peaks.
        '''
        if self.energy_filtered is None:
            raise ValueError("Background is not calculated. Run get_background() method first.")
        if self.element == 'carbon':
            self.lmfit_model(peaks_model=model, fixed_mix=fixed_mix, mix=mix)
            self.peak_fit_result = casa_fit_peaks(self.intensity_filtered, self.energy_filtered, self.model, self.model_params, method=method)
        elif self.element == 'oxygen' and (model == '2peaks' or model == '3peaks' or model == '4peaks'):
            self.lmfit_model(peaks_model=model)
            self.peak_fit_result = fit_oxygen_peaks(self.intensity_filtered, self.energy_filtered, self.model, self.model_params)
        elif self.element == 'oxygen' and model == '6peaks_la':
            self.model, self.peak_fit_result, self.model_params, self.peaks_model = find_best_oxygen_fit(self.intensity_filtered, self.energy_filtered, self.peak_config)
    
    def plot_peak_fit(self, plot=True):
        '''
        Plots the peak fit.
        '''
        if self.peak_fit_result is None:
            raise ValueError("Peak fit is not calculated. Run fit_peaks() method first.")
        if plot:
            plot_full_peak_fit(self.peak_fit_result, self.energy, self.intensity, self.background, model=self.peaks_model, element=self.element, xps_config=self.peak_config, plot=plot)
        if not plot:
            plot_fig, plot_ax = plot_full_peak_fit(self.peak_fit_result, self.energy, self.intensity, self.background, model=self.peaks_model, element=self.element, xps_config=self.peak_config, plot=plot)
            return plot_fig, plot_ax

    def rsd(self):
        '''
        Calculates the RSD.
        '''
        if self.peak_fit_result is None:
            raise ValueError("Peak fit is not calculated. Run fit_peaks() method first.")
        # intensity_corrected = self.intensity_filtered - self.background[1]
        return calculate_rsd(self.intensity_filtered + self.background[1], self.peak_fit_result.best_fit + self.background[1])
    
    def get_peak_table(self, save_path=None):
        '''
        Returns the peak table.
        '''
        if self.peak_fit_result is None:
            raise ValueError("Peak fit is not calculated. Run fit_peaks() method first.")
        peak_table = extract_fit_results(self.element, self.peaks_model, self.peak_fit_result, self.peak_config)
        if save_path:
            peak_table.to_csv(save_path, index=False)
        self.peak_table = peak_table
        return peak_table


class SampleXPS():
    def __init__(self, carbon_path=None, oxygen_path=None, survey_path=None, peak_config=None):
        self.carbon_path = carbon_path
        self.oxygen_path = oxygen_path
        self.survey_path = survey_path
        self.peak_config = peak_config
        self.carbon_results = None
        self.oxygen_results = None
        self.elemental_composition = None
        pass
    
    def load_data(self, carbon_path=None, oxygen_path=None, survey_path=None, peak_config=None):
        '''
        Loads the data.
        '''
        if carbon_path:
            self.carbon_path = carbon_path
        if oxygen_path:
            self.oxygen_path = oxygen_path
        if survey_path:
            self.survey_path = survey_path
        if peak_config:
            self.peak_config = peak_config
        if self.carbon_path:
            self.carbon_data = XPSData('carbon', *extract_energy_intensity(self.carbon_path), peak_config=self.peak_config)
        if self.oxygen_path:
            self.oxygen_data = XPSData('oxygen', *extract_energy_intensity(self.oxygen_path), peak_config=self.peak_config)
        if self.survey_path:
            self.survey_data = XPSData('survey', *extract_energy_intensity(self.survey_path), peak_config=self.peak_config)
    
    def auto_carbon_fit(self, start_energy=295, end_energy=280, model='6peaks_la', method='least_squares', fixed_mix=False, mix=0.3):
        '''
        Automatically fits the carbon peaks.
        '''
        if self.carbon_data is None:
            raise ValueError("Carbon data is not loaded. Run load_data() method first.")
        self.carbon_data.get_background(start_energy, end_energy)
        self.carbon_data.fit_peaks(model=model, method=method, fixed_mix=fixed_mix, mix=mix)
        self.carbon_results = self.carbon_data.peak_fit_result

    def auto_oxygen_fit(self, start_energy=542, end_energy=526, model='6peaks_la'):
        '''
        Automatically fits the oxygen peaks.
        '''
        if self.oxygen_data is None:
            raise ValueError("Oxygen data is not loaded. Run load_data() method first.")
        self.oxygen_data.get_background(start_energy, end_energy)
        self.oxygen_data.fit_peaks(model=model)
        self.oxygen_results = self.oxygen_data.peak_fit_result

    def get_composition(self):
        '''
        Returns the composition.
        '''
        if self.carbon_results is None or self.oxygen_results is None:
            raise ValueError("Carbon or oxygen fits are not calculated. Run auto_carbon_fit() and auto_oxygen_fit() methods first.")
        carbon_df = self.carbon_data.get_peak_table()
        oxygen_df = self.oxygen_data.get_peak_table()
        elemental_composition = calculate_ratios(carbon_df, oxygen_df)
        self.elemental_composition = elemental_composition
        return elemental_composition
    
    def run_analysis(self):
        '''
        Runs the full analysis.
        '''
        if self.oxygen_path is None or self.carbon_path is None:
            raise ValueError("Carbon or oxygen data is not loaded. Run load_data() method first.")
        self.load_data(carbon_path=self.carbon_path, oxygen_path=self.oxygen_path, survey_path=self.survey_path, peak_config=self.peak_config)
        self.auto_carbon_fit()
        self.auto_oxygen_fit()
        self.get_composition()
        self.carbon_data.plot_peak_fit()
        self.oxygen_data.plot_peak_fit()
        # carbon_fig, carbon_ax = self.carbon_data.plot_peak_fit(plot=False)
        # oxygen_fig, oxygen_ax = self.oxygen_data.plot_peak_fit(plot=False)
        # self.oxygen_fig = full_peak_with_table_fig(oxygen_fig, oxygen_ax, self.oxygen_data.peak_table, plot=True)
        # self.carbon_fig = full_peak_with_table_fig(carbon_fig, carbon_ax, self.carbon_data.peak_table, plot=True)
        print(self.elemental_composition)

    def save_analysis(self, save_path):
        '''
        Saves the analysis.
        '''
        if self.elemental_composition is None:
            raise ValueError("Analysis is not run. Run run_analysis() method first.")
        # save carbon and oxygen lmfit results
        with open(save_path+'carbon_model_results.pkl', 'wb') as file:
            pickle.dump(self.carbon_results, file)
        with open(save_path+'oxygen_model_results.pkl', 'wb') as file:
            pickle.dump(self.oxygen_results, file)
        # save carbon and oxygen peak plots with tables
        # self.carbon_fig.savefig(save_path+'carbon_peak_fit.png', dpi=300)
        # self.oxygen_fig.savefig(save_path+'oxygen_peak_fit.png', dpi=300)
        # save elemental composition
        with open(save_path+'composition.txt', 'w') as file:
            file.write(str(self.elemental_composition))