import matplotlib.pyplot as plt
import numpy as np
from hc_xps.peak_fit import get_peaks_config, calculate_rsd
import tomllib
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import polars as pl


def plot_basic_xps(energy, intensity):
    plt.plot(energy, intensity)
    plt.xlabel("Binding Energy (eV)")
    plt.ylabel("Intensity (a.u.)")
    plt.gca().invert_xaxis()
    plt.show()


def plot_xps_with_background(energy, intensity, background):
    plt.plot(energy, intensity, label='XPS Data')
    plt.plot(background[0], background[1], label='Background')
    plt.xlabel("Binding Energy (eV)")
    plt.ylabel("Intensity (a.u.)")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show()


# def plot_full_peak_fit(result, energy, intensity, background, model='5peaks', element='carbon', xps_config=None):
#     comps = result.eval_components(x=result.userkws['x'])
#     fig, (ax_residuals, ax_xps) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [1, 4]})
#     if xps_config is None:
#         peaks_config = get_peaks_config()
#     else:
#         with open(xps_config, 'rb') as file:
#             peaks_config = tomllib.load(file)
#     peaks = peaks_config[element]['models'][model].split('+')
#     for peak in peaks:
#         ax_xps.plot(result.userkws['x'], comps[f'{peak}_']+background[1], linestyle='--', label=peaks_config[element]['peaks'][peak]['docstring'].strip())
#     ax_xps.scatter(energy, intensity, label='XPS Data', s=5, color='black')
#     # ax_xps.scatter(result.userkws['x'], result.data, label='XPS Data', s=5, color='black')
#     # ax_xps.plot(background[0], result.best_fit+background[1], label='Fit', linestyle='solid')
#     # ax_xps.plot(background[0], background[1], label='Background')
#     ax_xps.plot(result.userkws['x'], result.best_fit+background[1], label='Fit', linestyle='solid')
#     ax_xps.plot(result.userkws['x'], background[1], label='Background')
#     ax_xps.legend()
#     ax_xps.set_xlabel("Binding Energy (eV)")
#     ax_xps.set_ylabel("Intensity (a.u.)")
#     ax_xps.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
#     ax_xps.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
#     ax_xps.yaxis.get_offset_text().set_fontsize(10)
#     ax_xps.invert_xaxis()
#     # start_idx = np.argmin(abs(energy - background[0][0]))  # Getting the start and end positions for the intensity envelope
#     # end_idx = np.argmin(abs(energy - background[0][-1]))
#     # intensity_filtered = intensity[start_idx:end_idx+1]
#     # rsd = calculate_rsd(intensity_filtered, result.best_fit+background[1])   # Calculating the RSD using the original data and the best fit witht the added background
#     rsd = calculate_rsd(result.data + background[1], result.best_fit + background[1])
#     # ax_residuals.plot(background[0], result.residual, color='black')
#     ax_residuals.plot(result.userkws['x'], result.residual, color='black')
#     ax_residuals.axhline(0, color='gray', linestyle='--')
#     ax_residuals.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
#     ax_residuals.get_yaxis().set_visible(False)
#     ax_xps.spines['top'].set_visible(False)
#     ax_residuals.spines['bottom'].set_visible(False)
#     ax_residuals.text(0.9, 0.2, f'RSD: {rsd:.2f}', transform=ax_residuals.transAxes, fontsize=12, ha='center')
#     ax_residuals.invert_xaxis()
#     fig.subplots_adjust(hspace=0)
#     plt.gca().invert_xaxis()
#     plt.show()


def plot_full_peak_fit(result, energy, intensity, background, model='5peaks', element='carbon', xps_config=None, plot=True):
    comps = result.eval_components(x=result.userkws['x'])
    if xps_config is None:
        peaks_config = get_peaks_config()
    else:
        with open(xps_config, 'rb') as file:
            peaks_config = tomllib.load(file)
    peaks = peaks_config[element]['models'][model].split('+')
    fig, ax_main = plt.subplots(figsize=(8, 6))
    ax_resid = ax_main.inset_axes([0, 0.85, 1, 0.15], sharex=ax_main)
    for peak in peaks:
        ax_main.plot(result.userkws['x'], comps[f'{peak}_']+background[1], linestyle='--', label=peaks_config[element]['peaks'][peak]['docstring'].strip())
    ax_main.scatter(energy, intensity, label='XPS Data', s=5, color='black')
    ax_main.plot(result.userkws['x'], result.best_fit+background[1], label='Fit', linestyle='solid')
    ax_main.plot(result.userkws['x'], background[1], label='Background')
    ax_main.set_ylim(top=1.2*intensity.max())
    ax_main.plot([], [], 'k-', label="Residuals")
    ax_main.legend(loc="center left", fontsize=10)
    # legend = ax_main.legend(loc="upper left", fontsize=10)
    # legend.get_frame().set_facecolor("white")  # Set background to white
    # legend.get_frame().set_edgecolor("black")  # Set edge to black
    # legend.get_frame().set_alpha(1.0)  # Fully opaque (no transparency)
    ax_main.set_xlabel("Binding Energy (eV)", fontsize=12)
    ax_main.set_ylabel("Intensity (a.u.), Resdiuals x 2", fontsize=12)
    ax_main.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax_main.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax_main.yaxis.get_offset_text().set_fontsize(10)
    ax_main.tick_params(axis='both', which='major', labelsize=10)
    ax_main.invert_xaxis()
    rsd = calculate_rsd(result.data + background[1], result.best_fit + background[1])
    ax_resid.plot(result.userkws['x'], result.residual*2, color='black')
    ax_resid.axhline(0, color='gray', linestyle='--')
    ax_resid.text(0.9, 0.0, f'RSD: {rsd:.2f}', transform=ax_resid.transAxes, fontsize=12, ha='center')
    ax_resid.axis('off')
    ax_resid.spines["top"].set_visible(False)
    ax_resid.spines["right"].set_visible(False)
    ax_resid.spines["left"].set_visible(False)
    ax_resid.set_ylim(-12500, 12500)
    ax_resid.invert_xaxis()
    if element == 'carbon':
        plt.title("C1s Scan")
    elif element == 'oxygen':
        plt.title("O1s Scan")
    plt.gca().invert_xaxis()
    if plot:
        plt.show()
    else:
        return fig, [ax_main, ax_resid]


def full_peak_with_table_fig(fig, axes, peak_table, plot=False):  # needs fixing to have normal looking plot
    rounded_df = peak_table.with_columns([
        pl.col(col_name).round(2) for col_name in peak_table.columns if peak_table[col_name].dtype in [pl.Float32, pl.Float64]
    ])
    fig.set_size_inches(16, 10)
    gs = GridSpec(2, 1, height_ratios=[4, 2], figure=fig)
    main_ax = axes[0]
    main_ax.set_position(gs[0].get_position(fig))
    main_ax.set_subplotspec(gs[0])
    resid_ax = axes[1]
    # resid_ax.set_position(
    #     # main_ax.get_position().x0 + 0.0 * main_ax.get_position().width,
    #     # main_ax.get_position().y0 + 0.85 * main_ax.get_position().height,
    #     1.0 * main_ax.get_position().width,
    #     0.15 * main_ax.get_position().height
    # )
    resid_ax.set_position(gs[0].get_position(fig))
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    table = ax_table.table(cellText=rounded_df.to_numpy(), colLabels=rounded_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(peak_table.columns))))
    if plot:
        plt.show()
    return fig


# def casa_plot_full_peak_fit():
#     comps = result.eval_components(x=background[0])
#     fig, (ax_residuals, ax_xps) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [1, 4]})