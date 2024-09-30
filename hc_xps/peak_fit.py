import os
import numpy as np
import polars as pl
import tomllib
from lmfit.models import VoigtModel


def get_peaks_config():
    with open(
        os.path.join(os.path.dirname(__file__), "spectrum_config/peaks_config.toml"),
        "rb",
    ) as file:
        peaks_config = tomllib.load(file)
    return peaks_config


def build_lmfit_model(model='5peaks', element='carbon', fixed_peaks=None, fixed_mix=None, mix=None):
    peaks_config = get_peaks_config()
    peaks = peaks_config[element]['models'][model].split('+')
    model_list = []
    for peak in peaks:
        peak_model = eval(peaks_config['element']['peaks'][peak]['peak_type']+f'Model(prefix="{peak}_")')
        for hint in peaks_config[element]['peaks'][peak]['param_hints']:
            peak_model.set_param_hint(f'{peak}_{hint}', **peaks_config[element]['peaks'][peak]['param_hints'][hint])
        model_list.append(peak_model)
    model = model_list[0]
    for i in range(1, len(model_list)):
        model += model_list[i]
    params = model.make_params()
    if mix is not None:
        params['mix'].set(value=mix, vary=True)
        for peak in peaks:
            params[f'{peak}_gamma'].set(expr=f'(mix/(1-mix))*sqrt(2*log(2))*{peak}_sigma')
    if fixed_mix is not None:
        params['mix'].set(value=fixed_mix, vary=False)
        for peak in peaks:
            params[f'{peak}_gamma'].set(expr=f'(mix/(1-mix))*sqrt(2*log(2))*{peak}_sigma')
    if fixed_peaks is not None:
        for peak in fixed_peaks:
            params[f'{peak}_sigma'].set(expr=f'A_sigma')
            params[f'{peak}_gamma'].set(expr=f'A_gamma')

    return model, params
    

def fit_peaks(intensity, energy, model='5peaks', element='carbon', fixed_peaks=None, fixed_mix=None, mix=None, method='powell'):
    model, params = build_lmfit_model(model, element, fixed_peaks, fixed_mix, mix)
    result = model.fit(data=intensity, params=params, x=energy, method=method)
    return result