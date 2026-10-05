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
SUMMARY_PATH = FIGS_DIR / 'summary.txt'

sys.path.append(str(SRC_DIR))

# The matplotlib backend must be selected before plot.py imports pyplot. Default to
# the non-interactive Agg backend (plt.show() becomes a no-op) unless --show is passed,
# so batch runs don't block waiting for a window to be closed.
if '--show' not in sys.argv:
    import matplotlib
    matplotlib.use('Agg')

import os
import numpy as np

from model import Model
from plot import RasterModel, summarize_peak_embeddings, save_peak_summary_txt

# multiprocessing.cpu_count() (used as the default `num_processes` inside plot.py)
# reports the machine's total core count, which can be well above what a cgroup/
# SLURM allocation actually grants this process — sched_getaffinity reports the
# real usable count, so pass it explicitly to avoid oversubscribing.
try:
    NUM_PROCESSES = len(os.sched_getaffinity(0))
except AttributeError:
    NUM_PROCESSES = os.cpu_count()


def _timed(label, fn):
    start = time.time()
    result = fn()
    print(f"\n{label} total execution time: {time.time() - start:.2f} seconds")
    return result


def run_fig1():
    """Fig 1.6: true prevalence vs germination_time, 3 maturity rates x 2 Bj panels,
    using the model's own default parameters (not the Silene/Dianthus species overrides).

    Uses eval=7 (see model.py Model.evaluate()), the actual infected-fraction-of-
    population prevalence metric, averaged over the periodic attractor. The previous
    version of this figure hardcoded eval=6 (genotype 1's share of two genotypes) via
    run_single_simulation(), which is meaningless for a single-strain sweep like this
    one (genotype 2 is never seeded) and degenerated to exactly 1.0 everywhere.
    """
    param1_values = np.linspace(0, 365, 15)
    param2_values = np.linspace(0, 365, 15)

    model = Model(X_0=(5, 0, 0, 5, 0, 0, 0, 0, 0, 0), t=(0, 365 * 400), eval=7)
    raster_model = RasterModel(model, 'infected_time', 'infected_time2', param1_values, param2_values)

    print("BetaJ: ", model.Bj * 365)
    print("BetaA: ", model.Bv * 365)

    def _run():
        return raster_model.plot_fig1(
            germination_values=np.linspace(1, 364, 30),
            # Default parameters (not Silene/Dianthus species overrides):
            maturity_values=[4 / 365, 8 / 365, 16 / 365],
            bj_values=[0.15 / 365, 0.4 / 365],
            maturity_labels=["Slow Maturation (4 yr$^{-1}$)", "Intermediate Maturation(8 yr$^{-1}$)",
                              "Fast Maturation(16 yr$^{-1}$)"],
            bj_labels=[r"$\beta_J$ = 0.15 yr$^{-1}$", r"$\beta_J$ = 0.4 yr$^{-1}$"],
            save_path=str(FIGS_DIR / "fig1.6.pdf"),
            results_save_path=str(FIGS_DIR / "fig1.6_embedding.npz"),
            use_saved_embedding=True,  # reuse the cache instead of resweeping if it matches
            num_processes=NUM_PROCESSES,
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

    # X_0 must seed BOTH genotypes' infected classes (indices 3-5 and 7-9). With
    # genotype 2 (Ij2/Iv2/If2) starting at 0, dIj2 = Bj*Sj*If2 - ... etc. are all
    # proportional to the already-zero genotype-2 populations, so it can never
    # establish and every ESS sweep point degenerates to "genotype 1 always wins"
    # regardless of germination/maturity. Seed symmetrically, matching Fig 4.
    model = Model(X_0=(5, 0, 0, 2.5, 0, 0, 0, 2.5, 0, 0), t=(0, 365 * 400),
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
            save_path=str(FIGS_DIR / "fig3.pdf"),
            results_save_path=str(FIGS_DIR / "fig3_embedding.npz"),
            num_processes=NUM_PROCESSES,
        )

    result = _timed("Fig 3", _run)
    save_peak_summary_txt(SUMMARY_PATH, "Fig 3", raster_model.fig3_peak_info,
                           x_label='germination day (ESS peak)')
    return result


def run_fig3_1x():
    """Copy of Fig 3 using maturity_multipliers=(1,2,4,8) instead of (4,8,16,32).

    Against the same base_maturity=4/365 (4 yr⁻¹), this gives absolute maturation
    rates of 4/8/16/32 yr⁻¹ — i.e. the legend now reads "1x, 2x, 4x, 8x" while the
    underlying rates match the 4/8/16 yr⁻¹ convention used elsewhere (e.g. Fig 1),
    instead of the original's 16/32/64/128 yr⁻¹.
    """
    param1_values = np.linspace(0, 365, 15)
    param2_values = np.linspace(0, 365, 15)

    model = Model(X_0=(5, 0, 0, 2.5, 0, 0, 0, 2.5, 0, 0), t=(0, 365 * 400),
                  germination_time=212.5, Bj=0.2 / 365, infected_time=182.5, infected_time2=182.5)
    raster_model = RasterModel(model, 'infected_time', 'infected_time2', param1_values, param2_values)

    print("BetaJ: ", model.Bj * 365)
    print("BetaA: ", model.Bv * 365)

    def _run():
        return raster_model.plot_fig3(
            germination_values=np.linspace(62.5, 302.5, 9),
            flowering_values=np.linspace(0, 365, 40),
            base_maturity=4 / 365,
            maturity_multipliers=(1, 2, 4, 8),
            save_path=str(FIGS_DIR / "fig3_1x.pdf"),
            results_save_path=str(FIGS_DIR / "fig3_1x_embedding.npz"),
            num_processes=NUM_PROCESSES,
        )

    result = _timed("Fig 3 (1x-8x)", _run)
    save_peak_summary_txt(SUMMARY_PATH, "Fig 3 (1x-8x)", raster_model.fig3_peak_info,
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


def run_fig4_invasion():
    """Fig 4 (true invasion analysis): same germination/config sweep as run_fig4(),
    but the ESS at each point comes from plot_fig4_invasion() — equilibrate the
    resident alone, then introduce every candidate mutant strategy at low frequency
    and see whether it grows or decays while rare — instead of the symmetric
    co-invasion approximation _compute_ess() uses.
    """
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
        return raster.plot_fig4_invasion(
            germination_values=germination_values,
            flowering_values=flowering_values,
            model_configs=model_configs,
            config_labels=['Basic', 'Silene latifolia', 'Dianthus pavonius'],
            results_save_path=str(FIGS_DIR / "fig4_invasion_embedding.npz"),  # resumes automatically if present
            save_path=str(FIGS_DIR / "fig4_invasion.pdf"),
            num_processes=NUM_PROCESSES,
        )

    result = _timed("Fig 4 (true invasion)", _run)
    save_peak_summary_txt(SUMMARY_PATH, "Fig 4 (true invasion)", raster.fig4_invasion_peak_info,
                           x_label='germination day (ESS peak)')
    return result


def run_fig3_2():
    """Fig 3.2: same 1x/2x/4x/8x maturity sweep as run_fig3_1x(), but using the
    long-run invasion analysis (plot_fig3_2): equilibrate the resident 400 years,
    introduce a rare mutant, run 400 MORE years, and read off who holds the
    population at the end — instead of either the symmetric co-invasion
    approximation (_compute_ess) or the short-window growth-rate regression
    (_compute_ess_invasion).
    """
    model = Model(X_0=(5, 0, 0, 2.5, 0, 0, 0, 2.5, 0, 0), t=(0, 365 * 400),
                  germination_time=212.5, Bj=0.2 / 365, infected_time=182.5, infected_time2=182.5)
    raster_model = RasterModel(model, 'infected_time', 'infected_time2', None, None)

    print("BetaJ: ", model.Bj * 365)
    print("BetaA: ", model.Bv * 365)

    def _run():
        return raster_model.plot_fig3_2(
            germination_values=np.linspace(62.5, 302.5, 9),
            flowering_values=np.linspace(0, 365, 40),
            base_maturity=4 / 365,
            maturity_multipliers=(1, 2, 4, 8),
            resident_years=400,
            invasion_years=400,
            save_path=str(FIGS_DIR / "fig3.2.pdf"),
            results_save_path=str(FIGS_DIR / "fig3.2_embedding.npz"),
            num_processes=NUM_PROCESSES,
        )

    result = _timed("Fig 3.2 (long invasion)", _run)
    save_peak_summary_txt(SUMMARY_PATH, "Fig 3.2 (long invasion)", raster_model.fig3_2_peak_info,
                           x_label='germination day (ESS peak)')
    return result


def run_fig4_2():
    """Fig 4.2: same germination/config sweep as run_fig4_invasion(), but using the
    long-run invasion analysis (plot_fig4_2) and reduced to just the single
    comparison panel, drawn with semi-transparent lines so coincident curves stay
    visible.
    """
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
        return raster.plot_fig4_2(
            germination_values=germination_values,
            flowering_values=flowering_values,
            model_configs=model_configs,
            config_labels=['Basic', 'Silene latifolia', 'Dianthus pavonius'],
            resident_years=400,
            invasion_years=400,
            line_alpha=0.65,
            results_save_path=str(FIGS_DIR / "fig4.2_embedding.npz"),  # resumes automatically if present
            save_path=str(FIGS_DIR / "fig4.2.pdf"),
            num_processes=NUM_PROCESSES,
        )

    result = _timed("Fig 4.2 (long invasion)", _run)
    save_peak_summary_txt(SUMMARY_PATH, "Fig 4.2 (long invasion)", raster.fig4_2_peak_info,
                           x_label='germination day (ESS peak)')
    return result


def run_fig5():
    """Fig 5: single stationary raster (no animation) of one long-run-invasion share
    matrix already computed for Fig 3.2 — default parameters (1x base maturity, i.e.
    4 yr^-1), germination day 302.5 (where every sweep so far has put the ESS peak).
    Reads directly from figs/fig3.2_embedding.npz; runs no new simulations.

    Annotates the full uninvadable BAND (horizontal shaded stripe), not a single
    point: this system usually has mutual founder control among a wide range of
    nearby strategies (verified directly — resident/mutant pairs within the band
    each win as resident against the other), not one sharp evolutionary optimum.
    A single vertical line (the earlier version of this figure) marks a different,
    non-ESS-relevant quantity — see plot_fig5_invasion_raster()'s docstring.
    """
    from plot import plot_fig5_invasion_raster

    band_lo, band_hi, _ = plot_fig5_invasion_raster(
        embedding_path=str(FIGS_DIR / "fig3.2_embedding.npz"),
        mult_label="1×",
        germination_time=302.5,
        save_path=str(FIGS_DIR / "fig5.pdf"),
    )
    print(f"\nFig 5: uninvadable band = [{band_lo:.2f}, {band_hi:.2f}]")
    return band_lo, band_hi


def run_fig3_3():
    """Fig 3.3: band version of Fig 3.2 — same 1x/2x/4x/8x maturity sweep, rebuilt
    entirely from the existing figs/fig3.2_embedding.npz (no resimulation). Plots
    the full range of mutually-uninvadable resident strategies at each germination
    time as a shaded ribbon, instead of a single ESS point per line (see
    _uninvadable_band() for why: the "single point" Fig 3.2 reports is usually
    picked by ~1e-6 numerical noise among a wide plateau of equally-uninvadable
    strategies exhibiting real mutual founder control, not a unique optimum).
    Fig 3.2 itself is left untouched — this is an additional, complementary view.
    """
    from plot import plot_fig3_3_bands

    embedding_path = FIGS_DIR / "fig3.2_embedding.npz"
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"{embedding_path} not found — run `figs.py --fig 3_2` first (Fig 3.3 "
            "reuses its cached share matrices rather than resimulating).")

    return plot_fig3_3_bands(
        embedding_path=str(embedding_path),
        mult_labels=('1×', '2×', '4×', '8×'),
        save_path=str(FIGS_DIR / "fig3.3.pdf"),
    )


def run_fig4_3():
    """Fig 4.3: band version of Fig 4.2 — same germination/config sweep, rebuilt
    entirely from the existing figs/fig4.2_embedding.npz (no resimulation). Single
    comparison panel, one shaded ribbon per config instead of one ESS point per
    line. Fig 4.2 itself is left untouched — this is an additional, complementary
    view (see plot_fig3_3_bands()/​_uninvadable_band() for the rationale).
    """
    from plot import plot_fig4_3_bands

    embedding_path = FIGS_DIR / "fig4.2_embedding.npz"
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"{embedding_path} not found — run `figs.py --fig 4_2` first (Fig 4.3 "
            "reuses its cached share matrices rather than resimulating).")

    return plot_fig4_3_bands(
        embedding_path=str(embedding_path),
        config_labels=('Basic', 'Silene latifolia', 'Dianthus pavonius'),
        save_path=str(FIGS_DIR / "fig4.3.pdf"),
    )


FIG4_CONFIGS = {
    'Basic': {},  # Model defaults
    'Silene latifolia': {  # Slowinski et al. 2025
        'Bj': 0.6 / 365, 'Bv': 0.4 / 365, 'Bf': 0.4 / 365,
        'births': 8 / 365, 'gamma': 0.001, 'death': 0.36 / 365,
        'maturity': 0.85 / 365, 'g': 2 / 365,
    },
    'Dianthus pavonius': {  # Bruns et al. 2017
        'Bj': 3.38e-4, 'Bv': 3.38e-5, 'Bf': 3.38e-5,
        'births': 1.3 / 365, 'gamma': 1.5e-4, 'death': 0.187 / 365,
        'maturity': 8 / 365, 'g': 2 / 365,
    },
}


def _sweep_unless_cached(name, embedding_path, configs, germination_values):
    """Floquet sweeps take ~1h on 32 cores, so an existing embedding is reused
    (delete the .npz to force a recompute) — re-running only redraws the figure."""
    from floquet import compute_floquet_sweep

    if Path(embedding_path).exists():
        print(f"{name}: reusing {embedding_path} (delete it to recompute).")
        return
    _timed(f"{name} (Floquet invasion fitness)", lambda: compute_floquet_sweep(
        configs, germination_values, np.linspace(0, 365, 40), NUM_PROCESSES, embedding_path))


def _save_ess_summary(name, embedding_path):
    """Write this figure's ESS-vs-germination peak block into summary.txt."""
    from plot import compute_curve_peaks
    from floquet import safe_label

    with np.load(os.fspath(embedding_path), allow_pickle=False) as data:
        germ = np.asarray(data['germination_values'])
        labels = [str(l) for l in data['labels']]
        curves = [np.asarray(data[f'{safe_label(l)}_ess']) for l in labels]
    peak_info = compute_curve_peaks(germ, curves, labels)
    save_peak_summary_txt(SUMMARY_PATH, name, peak_info, x_label='germination day (latest ESS)')
    return peak_info


# The fig 3 series sweeps maturation as multiples of 4/365; the Model default is
# 8/365, so the stored "1×" is half the default. Legends show multiples of the default.
MATURITY_DISPLAY = {'1×': '0.5×', '2×': '1×', '4×': '2×', '8×': '4×'}


def run_fig3_4():
    """Fig 3.4: 1x/2x/4x/8x maturity sweep (as Fig 3.2), ESS from invasion fitness
    (Floquet exponent of the rare mutant) instead of share after a finite run."""
    from floquet import compute_floquet_sweep, model_kwargs_from, plot_floquet_ess

    base = Model(Bj=0.2 / 365)
    configs = {f"{m}×": model_kwargs_from(base, maturity=4 / 365 * m) for m in (1, 2, 4, 8)}
    embedding = FIGS_DIR / "fig3.4_embedding.npz"
    _sweep_unless_cached("Fig 3.4", embedding, configs, np.linspace(62.5, 302.5, 9))
    result = plot_floquet_ess(embedding, legend_fmt="$m$ = {} default",
                              label_map=MATURITY_DISPLAY,
                              save_path=str(FIGS_DIR / "fig3.4.pdf"))
    _save_ess_summary("Fig 3.4 (invasion fitness)", embedding)
    return result


def run_fig4_4():
    """Fig 4.4: Basic / Silene / Dianthus sweep (as Fig 4.2, single comparison
    panel), ESS from invasion fitness (Floquet exponent)."""
    from floquet import compute_floquet_sweep, model_kwargs_from, plot_floquet_ess

    configs = {label: model_kwargs_from(Model(**over)) for label, over in FIG4_CONFIGS.items()}
    embedding = FIGS_DIR / "fig4.4_embedding.npz"
    _sweep_unless_cached("Fig 4.4", embedding, configs, np.linspace(62.5, 302.5, 18))
    result = plot_floquet_ess(embedding, line_alpha=0.65, figsize=(8, 6),
                              save_path=str(FIGS_DIR / "fig4.4.pdf"))
    _save_ess_summary("Fig 4.4 (invasion fitness)", embedding)
    return result


def run_fig5_2():
    """Fig 5.2: pairwise invasibility plot (40x40 invasion-fitness raster) at default
    maturation (the stored "2×" row), germination day 302.5, from the Fig 3.4 embedding."""
    from floquet import plot_floquet_pip

    return plot_floquet_pip(FIGS_DIR / "fig3.4_embedding.npz", "2×", 302.5,
                            display_label="1× default maturation",
                            save_path=str(FIGS_DIR / "fig5.2.pdf"))


def run_fig3_5():
    """Fig 3.5: Fig 3.4 redrawn along the convergence-stable branch, so the curve is
    unbroken; crosses mark germination times with no globally uninvadable strategy."""
    from floquet import plot_floquet_ess

    return plot_floquet_ess(
        FIGS_DIR / "fig3.4_embedding.npz", legend_fmt="$m$ = {} default",
        label_map=MATURITY_DISPLAY, connect_css=True, show_coexistence=True,
        title="Convergence-Stable Infectious Flowering Time\nvs Germination Timing"
              "\n(invasion fitness, Floquet exponent)",
        save_path=str(FIGS_DIR / "fig3.5.pdf"))


def run_fig4_5():
    """Fig 4.5: Fig 4.4 redrawn along the convergence-stable branch (see run_fig3_5)."""
    from floquet import plot_floquet_ess

    return plot_floquet_ess(
        FIGS_DIR / "fig4.4_embedding.npz", line_alpha=0.65, figsize=(8, 6),
        connect_css=True, show_coexistence=True,
        title="Convergence-Stable Infectious Flowering Time\nvs Germination Timing"
              "\n(invasion fitness, Floquet exponent)",
        save_path=str(FIGS_DIR / "fig4.5.pdf"))


def run_fig6():
    """Fig 6: pairwise invasibility raster at default maturation, germination day 119 —
    germination day 119 (no global ESS: red cells whose mirror image across the diagonal
    is also red are pairs that invade both ways, i.e. a protected dimorphism) beside day
    302.5 as the null, where every red cell has a blue mirror and a single ESS exists."""
    from floquet import plot_floquet_pip

    return plot_floquet_pip(FIGS_DIR / "fig4.4_embedding.npz", "Basic", [119.0, 302.5],
                            display_label="1× default maturation",
                            save_path=str(FIGS_DIR / "fig6.pdf"))


def run_fig7():
    """Fig 7: copy of Fig 6 recoloured by competitive outcome — red where genotype 1
    wins, blue where genotype 2 wins, white where they coexist. Antisymmetric by
    construction, so the polymorphic pairs read as the white patches off the diagonal."""
    from floquet import plot_floquet_pip

    return plot_floquet_pip(FIGS_DIR / "fig4.4_embedding.npz", "Basic", [119.0, 302.5],
                            display_label="1× default maturation", outcome=True,
                            save_path=str(FIGS_DIR / "fig7.pdf"))


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
    '3_1x': run_fig3_1x,
    '4': run_fig4,
    '4invasion': run_fig4_invasion,
    '3_2': run_fig3_2,
    '4_2': run_fig4_2,
    '5': run_fig5,
    '3_3': run_fig3_3,
    '4_3': run_fig4_3,
    '3_4': run_fig3_4,
    '4_4': run_fig4_4,
    '3_5': run_fig3_5,
    '4_5': run_fig4_5,
    '5_2': run_fig5_2,
    '6': run_fig6,
    '7': run_fig7,
    'summary': run_summary,
}


def main():
    parser = argparse.ArgumentParser(description="Generate Figures 1-4 (and a peak-day summary).")
    parser.add_argument('--fig', nargs='+', choices=sorted(FIGURES), default=['1', '2', '3', '4'],
                         help="Which figure(s) to run. Defaults to 1 2 3 4 (not summary/4invasion).")
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
