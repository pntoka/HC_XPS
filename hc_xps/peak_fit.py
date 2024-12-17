import os
import numpy as np
import polars as pl
import tomllib
from lmfit.models import VoigtModel, SkewedVoigtModel, Model
from hc_xps.functions import LA_conv, LA_peak


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
    if fixed_mix and mix is not None:
        params['B_mix'].set(value=mix, vary=False)
    return model, params


def casa_fit_peaks(intensity, energy, lmfit_model, params, method='least_squares'):
    if method == 'least_squares':
        result = lmfit_model.fit(data=intensity, params=params, x=energy, method=method, fit_kws={'ftol': 1e-10, 'xtol': 1e-10})
    else:
        result = lmfit_model.fit(data=intensity, params=params, x=energy, method=method)
    return result

# def casa_fit_peaks(intensity, energy, model='6peaks_la', element='carbon', fixed_peaks=None, fixed_mix=False, mix=None, method='least_squares', xps_config=None):
#     model, params = build_casa_lmfit_model(model, element, fixed_peaks, fixed_mix, mix, xps_config)
#     if method == 'least_squares':
#         result = model.fit(data=intensity, params=params, x=energy, method=method, fit_kws={'ftol': 1e-10, 'xtol': 1e-10})
#     else:
#         result = model.fit(data=intensity, params=params, x=energy, method=method)
#     return result


def fit_peaks(intensity, energy, model='5peaks', element='carbon', fixed_peaks=None, fixed_mix=None, mix=None, method='powell', xps_config=None):
    model, params = build_lmfit_model(model, element, fixed_peaks, fixed_mix, mix, xps_config)
    result = model.fit(data=intensity, params=params, x=energy, method=method)
    return result


def calculate_rsd(experimental, calculated):
    n = len(experimental)
    rsd = np.sqrt((1/n) * np.sum(((experimental - calculated) / np.sqrt(np.abs(experimental)))**2))
    return rsd