import re
import os
import numpy as np
import polars as pl


def parse_avg_file(file_path):
    '''Code from KherveFitting/libraries/Open.py'''
    with open(file_path, 'r') as file:
        content = file.read()
    photon_energy = float(re.search(r'DS_SOPROPID_ENERGY\s+:\s+VT_R4\s+=\s+(\d+\.\d+)', content).group(1))
    start_energy, width, num_points = map(float, re.search(r'\$SPACEAXES=1\s+0=\s+(\d+\.\d+),\s+(\d+\.\d+),\s+(\d+),',
                                                           content).groups())
    # Modified part to handle multiple numbers per line
    y_values = []
    for match in re.findall(r'LIST@\s+\d+=\s+([\d., ]+)', content):
        values = [float(val.strip()) for val in match.split(',')]
        y_values.extend(values)

    return photon_energy, start_energy, width, int(num_points), y_values


def extract_energy_intensity(file_path, to_csv=False):
    photon_energy, start_energy, width, num_points, y_values = parse_avg_file(file_path)
    be_values = [photon_energy - (start_energy + i * width) for i in range(num_points)]
    energy_intensity_data = np.array([be_values, y_values])
    if to_csv:
        np.savetxt(os.path.join(os.path.dirname(file_path), os.path.basename(file_path).replace('avg', 'csv')), energy_intensity_data.T, delimiter=',', header='Binding energy / eV,Counts',
                   comments='')
    return energy_intensity_data[0], energy_intensity_data[1]


def calculate_ratios(carbon_df, oxygen_df):
    total_carbon_area = carbon_df.filter(carbon_df['Peak ID'] != 'F')['Normalised Area'].sum()
    carbon_oxygen_peaks = ['C', 'D', 'E']
    total_carbon_oxygen_area = carbon_df.filter(pl.col('Peak ID').is_in(carbon_oxygen_peaks))['Normalised Area'].sum()
    total_oxygen_area = oxygen_df['Normalised Area'].sum()
    ratio_data = {
        'oxygen percentage': total_oxygen_area / (total_carbon_area + total_oxygen_area) * 100,
        'C-O to O ratio': total_carbon_oxygen_area / total_oxygen_area
    }
    return ratio_data