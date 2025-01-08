import os
import numpy as np
import polars as pl
import tomllib
from lmfit.models import VoigtModel, SkewedVoigtModel, Model
from hc_xps.functions import LA_conv, LA_peak, calculate_normalised_area, get_LA_peak_height, get_SV_peak_center_height


def get_peaks_config():
    with open(
        os.path.join(os.path.dirname(__file__), "spectrum_config/peaks_config.toml"),
        "rb",
    ) as file:
        peaks_config = tomllib.load(file)
    return peaks_config


def build_lmfit_model(model='5peaks', element='carbon', fixed_peaks=None, fixed_mix=None, mix=None, xps_config=None):
    if xps_config is None:
        peaks_config = get_peaks_config()
    else:
        with open(xps_config, 'rb') as file:
            peaks_config = tomllib.load(file)
    peaks = peaks_config[element]['models'][model].split('+')
    model_list = []
    for peak in peaks:
        if peak == 'Ela':
            peak_model = Model(LA_conv, prefix='Ela_')
        else:
            peak_model = eval(peaks_config[element]['peaks'][peak]['peak_type']+f'Model(prefix="{peak}_")')
        for hint in peaks_config[element]['peaks'][peak]['param_hints']:
            if (fixed_mix is None and mix is None) and (hint == 'mix') and peak != 'Ela':
                continue
            # if mix is None and hint == 'mix':
            #     continue
            peak_model.set_param_hint(f'{peak}_{hint}', **peaks_config[element]['peaks'][peak]['param_hints'][hint])
        model_list.append(peak_model)
    model = model_list[0]
    for i in range(1, len(model_list)):
        model += model_list[i]
    params = model.make_params()
    for peak in peaks:
        if peak == 'Ela':
            continue
        params[f'{peak}_gamma'].set(vary=True, expr='')
        if fixed_mix is not None or mix is not None:
            params[f'{peak}_mix'].set(vary=False)
    if mix is not None:
        for peak in peaks:
            params[f'{peak}_mix'].set(value=mix, vary=True)
            if peak == 'Ela':
                continue
            params[f'{peak}_gamma'].set(expr=f'({peak}_mix/(1-{peak}_mix))*sqrt(2*log(2))*{peak}_sigma')
    if fixed_mix is not None:
        for peak in peaks:
            if peak == 'Ela':
                continue
            params[f'{peak}_mix'].set(value=fixed_mix, vary=False)
            params[f'{peak}_gamma'].set(expr=f'({peak}_mix/(1-{peak}_mix))*sqrt(2*log(2))*{peak}_sigma')
    if fixed_peaks is not None:
        for peak in fixed_peaks:
            params[f'{peak}_sigma'].set(expr=f'{peaks[0]}_sigma')
            params[f'{peak}_gamma'].set(expr=f'{peaks[0]}_gamma')

    return model, params


def build_casa_lmfit_model(model='6peaks_la', element='carbon', fixed_peaks=None, fixed_mix=False, mix=None, xps_config=None):
    '''Makes the lmfit model for fitting carbon peaks according to fit from CasaXPS'''
    if xps_config is None:
        peaks_config = get_peaks_config()
    else:
        with open(xps_config, 'rb') as file:
            peaks_config = tomllib.load(file)
    peaks = peaks_config[element]['models'][model].split('+')
    model_list = []
    for peak in peaks:
        if peak == 'Ala':
            peak_model = Model(LA_peak, prefix='Ala_')
        elif peak == 'Alag':
            continue
        else:
            peak_model = eval(peaks_config[element]['peaks'][peak]['peak_type']+f'Model(prefix="{peak}_")')
        for hint in peaks_config[element]['peaks'][peak]['param_hints']:
            peak_model.set_param_hint(f'{peak}_{hint}', **peaks_config[element]['peaks'][peak]['param_hints'][hint])
        model_list.append(peak_model)
    model = model_list[0]
    for i in range(1, len(model_list)):
        model += model_list[i]
    params = model.make_params()
    for peak in peaks[1:]:
        params[f'{peak}_sp2_peak'].set(vary=True, expr=f'{peaks[0]}_center')
    params['F_amplitude'].set(expr=f'F_amp_ratio*{peaks[0]}_amplitude')
    if not fixed_mix and mix is None:
        for peak in peaks:
            if peak == 'Ala' or peak == 'Alag':
                continue
            params[f'{peak}_gamma'].set(vary=True, expr='')
    if fixed_peaks is not None:
        for peak in fixed_peaks:
            params[f'{peak}_sigma'].set(expr='B_sigma')
            params[f'{peak}_gamma'].set(expr='B_gamma')
    if mix is not None:
        params['B_mix'].set(value=mix, vary=True)
        params['B_gamma'].set(expr='(B_mix/(1-B_mix))*sqrt(2*log(2))*B_sigma')
        params['F_mix'].set(value=mix, vary=True)
        params['F_gamma'].set(expr='(F_mix/(1-F_mix))*sqrt(2*log(2))*F_sigma')
    if fixed_mix and mix is not None:
        params['B_mix'].set(value=mix, vary=False)
    return model, params


def casa_fit_peaks(intensity, energy, lmfit_model, params, method='least_squares'):
    if method == 'least_squares':
        result = lmfit_model.fit(data=intensity, params=params, x=energy, method=method, fit_kws={'ftol': 1e-9, 'xtol': 1e-9})
    else:
        result = lmfit_model.fit(data=intensity, params=params, x=energy, method=method)
    return result


def build_oxygen_lmfit_model(model='2peaks', xps_config=None):
    '''Makes the lmfit model for fitting oxygen peaks'''
    if xps_config is None:
        peaks_config = get_peaks_config()
    else:
        with open(xps_config, 'rb') as file:
            peaks_config = tomllib.load(file)
    peaks = peaks_config['oxygen']['models'][model].split('+')
    model_list = []
    for peak in peaks:
        peak_model = eval(peaks_config['oxygen']['peaks'][peak]['peak_type']+f'Model(prefix="{peak}_")')
        for hint in peaks_config['oxygen']['peaks'][peak]['param_hints']:
            peak_model.set_param_hint(f'{peak}_{hint}', **peaks_config['oxygen']['peaks'][peak]['param_hints'][hint])
        model_list.append(peak_model)
    model = model_list[0]
    for i in range(1, len(model_list)):
        model += model_list[i]
    params = model.make_params()
    for peak in peaks:
        params[f'{peak}_gamma'].set(vary=True, expr=f'({peak}_mix/(1-{peak}_mix))*sqrt(2*log(2))*{peak}_sigma')
    return model, params


def fit_oxygen_peaks(intensity, energy, lmfit_model, params, method='least_squares'):
    if method == 'least_squares':
        result = lmfit_model.fit(data=intensity, params=params, x=energy, method=method, fit_kws={'ftol': 1e-10, 'xtol': 1e-10})
    else:
        result = lmfit_model.fit(data=intensity, params=params, x=energy, method=method)
    return result


def find_best_oxygen_fit(intensity, energy, xps_config=None):
    if xps_config is None:
        peaks_config = get_peaks_config()
    else:
        with open(xps_config, 'rb') as file:
            peaks_config = tomllib.load(file)
    models = peaks_config['oxygen']['models']
    best_result = None
    best_model = None
    best_params = None
    best_peaks_model = None
    best_redchi = np.inf
    for model in models:
        lmfit_model, params = build_oxygen_lmfit_model(model, xps_config)
        result = fit_oxygen_peaks(intensity, energy, lmfit_model, params)
        redchi = result.redchi
        if redchi < best_redchi:
            best_redchi = redchi
            best_result = result
            best_model = lmfit_model
            best_params = params
            best_peaks_model = model
    return best_model, best_result, best_params, best_peaks_model


# def casa_fit_peaks(intensity, energy, model='6peaks_la', element='carbon', fixed_peaks=None, fixed_mix=False, mix=None, method='least_squares', xps_config=None):
#     model, params = build_casa_lmfit_model(model, element, fixed_peaks, fixed_mix, mix, xps_config)
#     if method == 'least_squares':
#         result = model.fit(data=intensity, params=params, x=energy, method=method, fit_kws={'ftol': 1e-10, 'xtol': 1e-10})
#     else:
#         result = model.fit(data=intensity, params=params, x=energy, method=method)
#     return result


# def fit_peaks(intensity, energy, model='5peaks', element='carbon', fixed_peaks=None, fixed_mix=None, mix=None, method='powell', xps_config=None):
#     '''Old fit peaks function for fitting carbon peaks'''
#     model, params = build_lmfit_model(model, element, fixed_peaks, fixed_mix, mix, xps_config)
#     result = model.fit(data=intensity, params=params, x=energy, method=method)
#     return result


def calculate_rsd(experimental, calculated):
    n = len(experimental)
    rsd = np.sqrt((1/n) * np.sum(((experimental - calculated) / np.sqrt(np.abs(experimental)))**2))
    return rsd


def calculate_lg_mix(sigma, gamma):
    lhm = 2*gamma
    ghm = 2*sigma*np.sqrt(2*np.log(2))
    mix = lhm/(lhm+ghm)
    return mix


def extract_fit_results(element, peak_model, fit_result, xps_config=None):
    if xps_config is None:
        peaks_config = get_peaks_config()
    else:
        with open(xps_config, 'rb') as file:
            peaks_config = tomllib.load(file)
    peaks = peaks_config[element]['models'][peak_model].split('+')
    rsf = peaks_config[element]['rsf']['rsf']
    data = {
        'Peak ID': [],
        'Peak name': [],
        'Position (eV)': [],
        'FWHM (eV)': [],
        'Height (CPS)': [],
        'L/G mix': [],
        'Area': [],
        'Normalised Area': [],
        'Peak type': []
    }
    for peak in peaks:
        data['Peak ID'].append(peak)
        data['Peak name'].append(peaks_config[element]['peaks'][peak]['docstring'])
        data['Position (eV)'].append(fit_result.params[f'{peak}_center'].value)
        data['FWHM (eV)'].append(fit_result.params[f'{peak}_fwhm'].value)
        if peak == 'Ala':
            ala_peak_height = get_LA_peak_height(
                fit_result.userkws['x'], fit_result.params['Ala_center'].value,
                fit_result.params['Ala_amplitude'].value,fit_result.params['Ala_fwhm'].value,
                fit_result.params['Ala_alpha'].value, fit_result.params['Ala_beta'].value)
            data['Height (CPS)'].append(ala_peak_height)
        if peak == 'Asv':
            asv_peak_center, asv_peak_height = get_SV_peak_center_height(
                fit_result.userkws['x'], fit_result.params['Asv_amplitude'].value,
                fit_result.params['Asv_center'].value, fit_result.params['Asv_sigma'].value,
                fit_result.params['Asv_gamma'].value, fit_result.params['Asv_skew'].value)
            data['Height (CPS)'].append(asv_peak_height)
        data['Height (CPS)'].append(fit_result.params[f'{peak}_height'].value)
        if peak == 'Ala' or peak == 'Asv':
            data['L/G mix'].append(np.nan)
        else:
            data['L/G ratio'].append(calculate_lg_mix(fit_result.params[f'{peak}_sigma'].value, fit_result.params[f'{peak}_gamma'].value))
        data['Area'].append(fit_result.params[f'{peak}_amplitude'].value)
        data['Normalised Area'].append(calculate_normalised_area(fit_result.params[f'{peak}_center'].value, fit_result.params[f'{peak}_amplitude'].value, rsf))
        data['Peak type'].append(peaks_config[element]['peaks'][peak]['peak_type'])
    df = pl.DataFrame(data)
    if 'Asv' in peaks:
        skew_column = [fit_result.params['Asv_skew'].value]
        skew_column.extend([np.nan for _ in range(len(peaks)-1)])
        actual_center = [asv_peak_center]
        actual_center.extend([np.nan for _ in range(len(peaks)-1)])
        df = df.with_column(pl.Series('Skew', skew_column))
        df = df.with_column(pl.Series('Actual SV Center', actual_center))
    if 'Ala' in peaks:
        alpha_column = [fit_result.params['Ala_alpha'].value]
        alpha_column.extend([np.nan for _ in range(len(peaks)-1)])
        beta_column = [fit_result.params['Ala_beta'].value]
        beta_column.extend([np.nan for _ in range(len(peaks)-1)])
        df = df.with_column(pl.Series('Alpha', alpha_column))
        df = df.with_column(pl.Series('Beta', beta_column))
    return df