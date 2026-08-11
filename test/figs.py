"""
Standalone script version of figs.ipynb — generates Figures 1-4.

Each figure is a self-contained function (run_fig1, run_fig2, run_fig3, run_fig4) so
any one of them can be run on its own, either from the command line or by importing
this module. fig1/fig2 reuse a saved embedding via use_saved_embedding=True, and fig4
resumes automatically from its checkpoint file whenever results_save_path already
exists on disk — so re-running a figure after an interruption (or to just rebuild the
plot) does not repeat already-computed simulations.

Each figure also writes/updates its own block (mean, std, and variance of the peak
day and peak value across its curves) in a shared summary.txt file. Re-running one
figure replaces only its block, leaving the others intact.

Usage:
    python figs.py                  # run all four figures + the peak summary
    python figs.py --fig 1 3        # run only figures 1 and 3
    python figs.py --fig summary    # just reload saved embeddings and print peak stats
    python figs.py --show           # also open each figure in an interactive window
    python figs.py --list           # list the available figure keys and exit
"""
import argparse
import sys
import time
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
REPO_DIR = TEST_DIR.parent
SRC_DIR = REPO_DIR / 'src'
FIGS_DIR = REPO_DIR / 'figs'
PUBFIGS_DIR = REPO_DIR.parent / 'PubFigs'
SUMMARY_PATH = FIGS_DIR / 'summary.txt'

sys.path.append(str(SRC_DIR))

# The matplotlib backend must be selected before plot.py imports pyplot. Default to
# the non-interactive Agg backend (plt.show() becomes a no-op) unless --show is passed,
# so batch runs don't block waiting for a window to be closed.
if '--show' not in sys.argv:
    import matplotlib
    matplotlib.use('Agg')

import numpy as np

from model import Model
from plot import RasterModel, summarize_peak_embeddings, save_peak_summary_txt


def _timed(label, fn):
    start = time.time()
    result = fn()
    print(f"\n{label} total execution time: {time.time() - start:.2f} seconds")
    return result


def run_fig1():
    """Fig 1: prevalence vs germination_time, 3 maturity rates x 2 Bj panels (Dianthus)."""
    param1_values = np.linspace(0, 365, 15)
    param2_values = np.linspace(0, 365, 15)

    model = Model(X_0=(5, 0, 0, 5, 0, 0, 0, 0, 0, 0), t=(0, 365 * 400))
    raster_model = RasterModel(model, 'infected_time', 'infected_time2', param1_values, param2_values)

    print("BetaJ: ", model.Bj * 365)
    print("BetaA: ", model.Bv * 365)

    def _run():
        return raster_model.plot_fig1(
            germination_values=np.linspace(1, 364, 30),
            # Dianthus:
            maturity_values=[4 / 365, 8 / 365, 16 / 365],
            bj_values=[3.38e-4, 6.76e-4],
            maturity_labels=["Slow Maturation (4 yr$^{-1}$)", "Intermediate Maturation(8 yr$^{-1}$)",
                              "Fast Maturation(16 yr$^{-1}$)"],
            bj_labels=[r"$\beta_J$ = 0.123 yr$^{-1}$", r"$\beta_J$ = 0.247 yr$^{-1}$"],
            save_path=str(FIGS_DIR / "fig1.6.Dianthus.pdf"),
            results_save_path=str(FIGS_DIR / "fig1.6.Dianthus_embedding.npz"),
            use_saved_embedding=True,  # reuse the cache instead of resweeping if it matches
        )

    result = _timed("Fig 1", _run)
    save_peak_summary_txt(SUMMARY_PATH, "Fig 1", raster_model.fig1_peak_info, x_label='germination day')
    return result


def run_fig2():
    """Fig 2: prevalence vs infected_time, 3 maturity rates x 3 germination panels (Dianthus)."""
    param1_values = np.linspace(0, 365, 15)
    param2_values = np.linspace(0, 365, 15)

    model = Model(X_0=(5, 0, 0, 5, 0, 0, 0, 0, 0, 0), t=(0, 365 * 800))
    raster_model = RasterModel(model, 'infected_time', 'infected_time2', param1_values, param2_values)

    print("BetaJ: ", model.Bj * 365)
    print("BetaA: ", model.Bv * 365)

    def _run():
        return raster_model.plot_fig2(
            infected_time_values=np.linspace(1, 364, 30),
            # Default + Dianthus:
            maturity_values=[4 / 365, 8 / 365, 16 / 365],
            maturity_labels=["Slow Maturation (2 yr$^{-1}$)", "Intermediate Maturation(4 yr$^{-1}$)",
                              "Fast Maturation(8 yr$^{-1}$)"],
            germination_time_values=[92.5, 182.5, 272.5],  # early, mid, late season
            germination_labels=["Early germination (day 92.5)",
                                 "Mid-season germination (day 182.5)",
                                 "Late germination (day 272.5)"],
            save_path=str(FIGS_DIR / "fig2.6.Dianthus.pdf"),
            results_save_path=str(FIGS_DIR / "fig2.6.Dianthus_embedding.npz"),
            use_saved_embedding=True,  # reuse the cache instead of resweeping if it matches
        )

    result = _timed("Fig 2", _run)
    save_peak_summary_txt(SUMMARY_PATH, "Fig 2", raster_model.fig2_peak_info, x_label='infected_time day')
    return result


def run_fig3():
    """Fig 3: ESS flowering time vs germination time for 4 maturity multipliers."""
    param1_values = np.linspace(0, 365, 15)
    param2_values = np.linspace(0, 365, 15)

    model = Model(X_0=(5, 0, 0, 5, 0, 0, 0, 0, 0, 0), t=(0, 365 * 400),
                  germination_time=212.5, Bj=0.2 / 365, infected_time=182.5, infected_time2=182.5)
    raster_model = RasterModel(model, 'infected_time', 'infected_time2', param1_values, param2_values)

    print("BetaJ: ", model.Bj * 365)
    print("BetaA: ", model.Bv * 365)

    def _run():
        return raster_model.plot_fig3(
            germination_values=np.linspace(62.5, 302.5, 9),
            flowering_values=np.linspace(0, 365, 40),
            base_maturity=4 / 365,
            maturity_multipliers=(4, 8, 16, 32),
            save_path=str(PUBFIGS_DIR / "fig3.pdf"),
        )

    result = _timed("Fig 3", _run)
    save_peak_summary_txt(SUMMARY_PATH, "Fig 3", raster_model.fig3_peak_info,
                           x_label='germination day (ESS peak)')
    return result


def run_fig4():
    """Fig 4: ESS flowering time vs germination time across model configs (Basic/Silene/Dianthus)."""
    model = Model(X_0=(5, 0, 0, 2.5, 0, 0, 0, 2.5, 0, 0), t=(0, 365 * 400))
    raster = RasterModel(model, 'infected_time', 'infected_time2', None, None)

    germination_values = np.linspace(62.5, 302.5, 18)  # x-axis points (18 values)
    flowering_values = np.linspace(0, 365, 40)  # raster grid (40x40 per point)

    model_configs = [
        {},  # basic — uses self.model defaults
        {  # Silene latifolia (Slowinski et al. 2025)
            'Bj': 0.6 / 365, 'Bv': 0.4 / 365, 'Bf': 0.4 / 365,
            'births': 8 / 365, 'gamma': 0.001, 'death': 0.36 / 365,
            'maturity': 0.85 / 365, 'g': 2 / 365,
        },
        {  # Dianthus pavonius (Bruns et al. 2017)
            'Bj': 3.38e-4, 'Bv': 3.38e-5, 'Bf': 3.38e-5,
            'births': 1.3 / 365, 'gamma': 1.5e-4, 'death': 0.187 / 365,
            'maturity': 8 / 365, 'g': 2 / 365,
        },
    ]

    def _run():
        return raster.plot_fig4(
            germination_values=germination_values,
            flowering_values=flowering_values,
            model_configs=model_configs,
            config_labels=['Basic', 'Silene latifolia', 'Dianthus pavonius'],
            results_save_path=str(FIGS_DIR / "fig4.6_embedding.npz"),  # resumes automatically if present
            save_path=str(FIGS_DIR / "fig4.pdf"),
        )

    result = _timed("Fig 4", _run)
    save_peak_summary_txt(SUMMARY_PATH, "Fig 4", raster.fig4_peak_info,
                           x_label='germination day (ESS peak)')
    return result


def run_summary():
    """Load the fig1/fig2/fig4 embeddings and print/plot combined peak-day statistics."""
    return summarize_peak_embeddings(
        paths=[
            str(FIGS_DIR / "fig1.6.Dianthus_embedding.npz"),
            str(FIGS_DIR / "fig2.6.Dianthus_embedding.npz"),
            str(FIGS_DIR / "fig4.6_embedding.npz"),
        ],
        source_labels=["Fig 1 (germination sweep)", "Fig 2 (infected_time sweep)", "Fig 4 (ESS sweep)"],
    )


FIGURES = {
    '1': run_fig1,
    '2': run_fig2,
    '3': run_fig3,
    '4': run_fig4,
    'summary': run_summary,
}


def main():
    parser = argparse.ArgumentParser(description="Generate Figures 1-4 (and a peak-day summary).")
    parser.add_argument('--fig', nargs='+', choices=sorted(FIGURES), default=['1', '2', '3', '4'],
                         help="Which figure(s) to run. Defaults to 1 2 3 4 (not summary).")
    parser.add_argument('--show', action='store_true',
                         help="Open each figure in an interactive window (blocks until closed). "
                              "Default is headless: figures are only written to disk.")
    parser.add_argument('--list', action='store_true', help="List available figure keys and exit.")
    args = parser.parse_args()

    if args.list:
        print("Available: " + ", ".join(sorted(FIGURES)))
        return

    for key in args.fig:
        FIGURES[key]()


if __name__ == '__main__':
    main()
