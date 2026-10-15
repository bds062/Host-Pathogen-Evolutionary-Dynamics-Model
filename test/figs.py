"""Reproduce the figures of the manuscript.

    python figs.py --list                      # the keys below
    python figs.py --fig sweeps                # all simulations (hours; see below)
    python figs.py --fig fig2 fig3 fig4 fig5 fig6   # draw the figures from saved sweeps
    python figs.py                             # everything

Keys
    prevalence     equilibrium-prevalence sweeps behind Figures 2 and 3 (minutes)
    maturation     invasion-fitness sweep, S. latifolia at 1x and 2x maturation (Figs 4, 5B)
    susceptibility invasion-fitness sweep, S. latifolia at 1.5:1 to 12:1 juvenile:adult
                   susceptibility (Fig 5A)
    species        invasion-fitness sweep, S. latifolia vs D. pavonius (Fig 6)
    sweeps         all of the above
    fig2 .. fig6   the figures; each needs its sweep (run automatically if missing)

Each invasion-fitness sweep takes about an hour on 32 cores and is cached in figs/ as an
.npz file (delete it to recompute). Everything is written to figs/, which is not tracked.
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / 'figs'
FIGS.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / 'src'))

from floquet import (_rc, compute_floquet_sweep, model_kwargs_from, plot_floquet_ess,  # noqa: E402
                     plot_floquet_pip)
from model import Model                                                                 # noqa: E402
from parameters import SILENE, SPECIES, X0                                              # noqa: E402
from plot import prevalence_vs_flowering, prevalence_vs_germination                     # noqa: E402

try:
    NUM_PROCESSES = len(os.sched_getaffinity(0))
except AttributeError:
    NUM_PROCESSES = os.cpu_count()

# ── design shared by all figures ───────────────────────────────────────────────
MATURATION_FACTORS = (1, 2)                       # estimated and twice the S. latifolia rate
SUSCEPTIBILITY_FACTORS = (1, 2, 4, 8)             # juvenile:adult ratio 1.5:1, 3:1, 6:1, 12:1
GERMINATION_MIDPOINTS = np.linspace(62.5, 302.5, 18)
FLOWERING_MIDPOINTS = np.linspace(0, 365, 40)
RASTER_GERMINATION = [119.0, 302.5]               # germination midpoints shown as rasters
BLUE, RED, ORANGE = '#2563eb', '#dc2626', '#d97706'
SUSCEPTIBILITY_COLORS = ['#2563eb', '#7c3aed', '#db2777', '#d97706']

EMBEDDINGS = {
    'prevalence_germination': FIGS / 'prevalence_germination.npz',
    'prevalence_flowering': FIGS / 'prevalence_flowering.npz',
    'maturation': FIGS / 'sweep_maturation.npz',
    'susceptibility': FIGS / 'sweep_susceptibility.npz',
    'species': FIGS / 'sweep_species.npz',
}


def m_label(factor):
    return f'm = {SILENE["maturity"] * factor:.4f}'


def ratio_label(factor):
    return f'{1.5 * factor:g}:1'


# ── simulations ────────────────────────────────────────────────────────────────
def _floquet_sweep(key, configs):
    path = EMBEDDINGS[key]
    if path.exists():
        print(f'{key}: reusing {path} (delete it to recompute)')
        return
    compute_floquet_sweep(configs, GERMINATION_MIDPOINTS, FLOWERING_MIDPOINTS, NUM_PROCESSES,
                          path)


def sweep_maturation():
    silene = Model(**SILENE)
    _floquet_sweep('maturation', {f'{f}×': model_kwargs_from(silene, maturity=SILENE['maturity'] * f)
                                  for f in MATURATION_FACTORS})


def sweep_susceptibility():
    silene = Model(**SILENE)
    _floquet_sweep('susceptibility', {f'{f}×': model_kwargs_from(silene, Bj=SILENE['Bj'] * f)
                                      for f in SUSCEPTIBILITY_FACTORS})


def sweep_species():
    _floquet_sweep('species', {name: model_kwargs_from(Model(**p)) for name, p in SPECIES.items()})


def sweep_prevalence():
    fig2()
    fig3()


# ── figures ────────────────────────────────────────────────────────────────────
def fig2():
    """Equilibrium prevalence against germination midpoint (three juvenile:adult
    susceptibility ratios, two maturation rates)."""
    bj_factors = (1, 2, 8)
    prevalence_vs_germination(
        base_kwargs=dict(SILENE, X_0=X0, t=(0, 365 * 400)),
        germination_values=np.linspace(1, 364, 30),
        maturity_values=[SILENE['maturity'] * f for f in MATURATION_FACTORS],
        bj_values=[SILENE['Bj'] * f for f in bj_factors],
        maturity_labels=[m_label(f) for f in MATURATION_FACTORS],
        bj_labels=[f'$\\beta_J$ = {SILENE["Bj"] * f:.4f} ({ratio_label(f)})' for f in bj_factors],
        colors=[BLUE, RED], save_path=FIGS / 'fig2.pdf',
        embedding_path=EMBEDDINGS['prevalence_germination'], num_processes=NUM_PROCESSES)


def fig3():
    """Equilibrium prevalence against infectious flowering midpoint at three germination
    midpoints, two maturation rates."""
    prevalence_vs_flowering(
        base_kwargs=dict(SILENE, X_0=X0, t=(0, 365 * 800)),
        flowering_values=np.linspace(1, 364, 30),
        maturity_values=[SILENE['maturity'] * f for f in MATURATION_FACTORS],
        germination_values=[92.5, 182.5, 272.5],
        maturity_labels=[m_label(f) for f in MATURATION_FACTORS],
        germination_labels=[f'Germination midpoint day {g:g}' for g in (92.5, 182.5, 272.5)],
        colors=[BLUE, RED], save_path=FIGS / 'fig3.pdf',
        embedding_path=EMBEDDINGS['prevalence_flowering'], num_processes=NUM_PROCESSES)


PIP_KW = dict(outcome=True, suptitle=False, xlabel='Genotype 1 flowering (day)',
              ylabel='Genotype 2 flowering (day)', panel_fmt='Germination midpoint {:.1f}',
              cbar_label='Genotype 1 advantage (per year)', strategies_in_title=True)
ESS_KW = dict(connect_css=True, show_coexistence=True, line_alpha=0.9, title=False,
              diag_label='1:1', xlabel='Germination midpoint (day)',
              ylabel='ESS infectious flowering (day)',
              coexist_labels=('No single ESS', 'Coexisting genotypes', ''))


def fig4():
    """Competitive outcome of every pair of infectious flowering times, S. latifolia, at
    germination midpoints 119 (A) and 302.5 (B)."""
    sweep_species()
    plot_floquet_pip(EMBEDDINGS['species'], 'Silene latifolia', RASTER_GERMINATION,
                     figsize=(10, 4.4), panel_letters=True, save_path=FIGS / 'fig4.pdf', **PIP_KW)


def fig5():
    """ESS infectious flowering midpoint against germination midpoint: (A) juvenile:adult
    susceptibility, (B) maturation rate."""
    sweep_susceptibility()
    sweep_maturation()
    _rc()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    kw = dict(ESS_KW, legend_loc='lower right')
    plot_floquet_ess(EMBEDDINGS['susceptibility'], labels=[f'{f}×' for f in SUSCEPTIBILITY_FACTORS],
                     colors=SUSCEPTIBILITY_COLORS,
                     label_map={f'{f}×': ratio_label(f) for f in SUSCEPTIBILITY_FACTORS},
                     ax=ax_a, **kw)
    plot_floquet_ess(EMBEDDINGS['maturation'], labels=[f'{f}×' for f in MATURATION_FACTORS],
                     colors=[BLUE, RED], label_map={f'{f}×': m_label(f) for f in MATURATION_FACTORS},
                     ax=ax_b, **kw)
    ax_b.set_ylabel('')
    for ax, letter, head in ((ax_a, 'A', 'Juvenile : adult susceptibility'),
                             (ax_b, 'B', 'Maturation rate (per day)')):
        ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=14, fontweight='bold',
                ha='right', va='bottom')
        ax.set_title(head, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig5.pdf', dpi=300, bbox_inches='tight')
    print(f'Figure saved to: {FIGS / "fig5.pdf"}')
    plt.close(fig)


def fig6():
    """S. latifolia versus D. pavonius: (A) ESS against germination midpoint for both
    species; (B) competitive-outcome rasters for D. pavonius at germination midpoints 119
    and 302.5."""
    sweep_species()
    _rc()
    fig = plt.figure(figsize=(10, 9.2))
    top, bottom = fig.subfigures(2, 1, height_ratios=(1, 1.05), hspace=0.08)
    ax = top.add_axes([0.26, 0.14, 0.5, 0.78])
    plot_floquet_ess(EMBEDDINGS['species'], labels=list(SPECIES), colors=[BLUE, ORANGE],
                     label_map={'Silene latifolia': '$S. latifolia$',
                                'Dianthus pavonius': '$D. pavonius$'}, ax=ax, **ESS_KW)
    axes = bottom.subplots(1, 2)
    plot_floquet_pip(EMBEDDINGS['species'], 'Dianthus pavonius', RASTER_GERMINATION,
                     fig=bottom, axes=axes, **PIP_KW)
    for sub, letter in ((top, 'A'), (bottom, 'B')):
        sub.text(0.02, 0.97, letter, fontsize=14, fontweight='bold', ha='left', va='top')
    fig.savefig(FIGS / 'fig6.pdf', dpi=300, bbox_inches='tight')
    print(f'Figure saved to: {FIGS / "fig6.pdf"}')
    plt.close(fig)


TASKS = {
    'prevalence': sweep_prevalence, 'maturation': sweep_maturation,
    'susceptibility': sweep_susceptibility, 'species': sweep_species,
    'sweeps': lambda: [sweep_prevalence(), sweep_maturation(), sweep_susceptibility(),
                       sweep_species()],
    'fig2': fig2, 'fig3': fig3, 'fig4': fig4, 'fig5': fig5, 'fig6': fig6,
}


def main():
    parser = argparse.ArgumentParser(description='Reproduce the figures of the manuscript.')
    parser.add_argument('--fig', nargs='+', choices=sorted(TASKS),
                        default=['fig2', 'fig3', 'fig4', 'fig5', 'fig6'])
    parser.add_argument('--list', action='store_true', help='list the keys and exit')
    args = parser.parse_args()
    if args.list:
        print('Available: ' + ', '.join(sorted(TASKS)))
        return
    for key in args.fig:
        TASKS[key]()


if __name__ == '__main__':
    main()
