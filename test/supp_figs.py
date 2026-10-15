"""Supplementary figures.

  S1  structure of the model (compartments and flows)
  S2  300-year run from the standard initial condition, showing the approach to the
      repeating annual cycle on which equilibrium prevalence is measured
  S3  realized germination for germination midpoints 62.5, 119 and 189.6, with infectious
      flowering at the convergence-stable strategy of the species sweep

    python supp_figs.py            -> figs/figS1.pdf, figS2.pdf, figS3.pdf (+ .png)

S3 reads figs/sweep_species.npz (python figs.py --fig species).
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch      # noqa: E402
from scipy.integrate import solve_ivp                               # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / 'figs'
FIGS.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / 'src'))
from floquet import _rc, css_track                                  # noqa: E402
from model import Model                                             # noqa: E402
from parameters import FLOWERING_MIDPOINT, SILENE, X0               # noqa: E402
from timing import germination, temp                                # noqa: E402

BLUE, PINK, GRAY, PURPLE = '#2563eb', '#db2777', '#9ca3af', '#6d28d9'
FLOWER_ON = (133.75, 231.25)      # healthy flowering window (rate above half its maximum)
EXAMPLE_GERMINATION = 119.0       # germination midpoint used for S2
S3_MIDPOINTS = (62.5, 119.0, 189.6)


def _model(germ, inf, years):
    return Model(X_0=X0, t=(0, 365 * years), germination_time=germ, infected_time=inf,
                 infected_time2=inf, **SILENE)


def _save(fig, name, **kw):
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'{name}.{ext}', dpi=300, bbox_inches='tight', **kw)
    print('saved', FIGS / f'{name}.pdf')
    plt.close(fig)


def _ess_for(germ):
    """Convergence-stable infectious flowering midpoint for S. latifolia at `germ`."""
    with np.load(FIGS / 'sweep_species.npz') as z:
        germs = z['germination_values']
        track, _ = css_track(germs, z['Silene_latifolia_sing_x'], z['Silene_latifolia_sing_kind'])
    return float(track[np.argmin(np.abs(germs - germ))])


# ── S1: model diagram ──────────────────────────────────────────────────────────
def fig_s1():
    rc = {'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
          'mathtext.fontset': 'stix'}
    ink, inf = 'black', '#4c1d95'
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(12, 6.4))
        ax.set_xlim(-0.3, 12.6); ax.set_ylim(-1.15, 6.15); ax.set_aspect('equal'); ax.axis('off')
        W, H = 1.9, 1.05
        SY, IY = 3.4, 0.55
        X = {'d': 0.2, 'j': 3.3, 'v': 6.4, 'f': 9.5}

        def box(x, y, lab, infected=False):
            ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle='round,pad=0,rounding_size=0.12',
                                        fc='#f3f0fa' if infected else 'white',
                                        ec=inf if infected else ink, lw=1.6, zorder=3))
            ax.text(x + W / 2, y + H / 2, lab, ha='center', va='center', fontsize=24, zorder=4,
                    color=inf if infected else ink)

        def flow(p0, p1, color=ink, rad=0.0, lw=1.5):
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=16,
                                         connectionstyle=f'arc3,rad={rad}', color=color, lw=lw,
                                         shrinkA=0, shrinkB=0, zorder=2))

        def label(x, y, t, color=ink, ha='center', size=19):
            ax.text(x, y, t, fontsize=size, color=color, ha=ha, va='center', zorder=5)

        for k, lab in (('d', '$S_d$'), ('j', '$S_j$'), ('v', '$S_v$'), ('f', '$S_f$')):
            box(X[k], SY, lab)
        for k, lab in (('j', '$I_j$'), ('v', '$I_v$'), ('f', '$I_f$')):
            box(X[k], IY, lab, True)
        mid = lambda a, b: (X[a] + W + X[b]) / 2
        flow((X['d'] + W, SY + H / 2), (X['j'], SY + H / 2))
        label(mid('d', 'j'), SY + H / 2 + 0.27, '$g(t)$')
        for y, c, sub in ((SY, ink, 'S'), (IY, inf, 'I')):
            flow((X['j'] + W, y + H / 2), (X['v'], y + H / 2), c)
            label(mid('j', 'v'), y + H / 2 + 0.27, '$m$', c)
            flow((X['v'] + W, y + 0.72 * H), (X['f'], y + 0.72 * H), c)
            flow((X['f'], y + 0.28 * H), (X['v'] + W, y + 0.28 * H), c)
            label(mid('v', 'f'), y + 0.72 * H + 0.27, f'$f_{sub}(t)$', c)
            label(mid('v', 'f'), y + 0.28 * H - 0.29, f'$v_{sub}(t)$', c)
        flow((X['f'] + 0.35 * W, SY + H), (X['d'] + 0.7 * W, SY + H), rad=0.2)
        label((X['f'] + X['d'] + W) / 2, 5.78, '$b\\,S_f$')
        for k, beta in (('j', 'j'), ('v', 'v'), ('f', 'f')):
            x = X[k] + 0.62 * W
            flow((x, SY), (x, IY + H), inf, lw=1.8)
            label(x + 0.12, (SY + IY + H) / 2, f'$\\beta_{beta}\\,I_f$', inf, ha='left')
        outs = [('d', SY + H, 1, '$\\mu/4$'), ('j', SY + H, 1, '$\\mu + \\gamma N$'),
                ('v', SY + H, 1, '$\\mu$'), ('f', SY + H, 1, '$\\mu$'),
                ('j', IY, -1, '$\\mu + \\gamma N$'), ('v', IY, -1, '$\\mu$'), ('f', IY, -1, '$\\mu$')]
        for k, y0, d, t in outs:
            x = X[k] + (0.2 if k == 'd' else 0.8) * W
            c = inf if d < 0 else ink
            flow((x, y0), (x, y0 + d * 0.5), c, lw=1.2)
            label(x + 0.1, y0 + d * 0.33, t, c, ha='left', size=16)
        ax.text(X['d'] + W / 2, IY + H / 2, 'Infected plants\nare sterile', ha='center',
                va='center', fontsize=16, color='#374151')
        ax.text(12.4, -1.0, '$N$ = total population', ha='right', fontsize=16, color='#374151')
        _save(fig, 'figS1', pad_inches=0.05)


# ── S2: approach to the annual cycle ───────────────────────────────────────────
def fig_s2(years=300, per=73):
    m = _model(EXAMPLE_GERMINATION, _ess_for(EXAMPLE_GERMINATION), years)
    t = np.linspace(0, 365 * years, per * years + 1)
    sol = solve_ivp(m.df, m.t, m.X_0, t_eval=t, rtol=1e-8, atol=1e-10, method='DOP853')
    Sj, Sv, Sf, Ij, Iv, If, Sd = sol.y[:7]
    yr = t / 365
    _rc()
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.4), sharex=True)
    for ax, y, c, lab in ((axes[0], Sd + Sj + Sv + Sf, BLUE, 'Healthy plants\n(density, a.u.)'),
                          (axes[1], Ij + Iv + If, PURPLE, 'Infected plants\n(density, a.u.)')):
        ax.plot(yr, y, color=c, lw=0.4, alpha=0.35)
        ax.plot(yr[::per], y[::per], color=c, lw=1.8)
        ax.set_ylabel(lab)
        ax.set_ylim(0, None)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[1].set_xlim(0, years)
    axes[1].set_xlabel('Years')
    handles = [plt.Line2D([], [], color='#6b7280', lw=5, alpha=0.35),
               plt.Line2D([], [], color='#111827', lw=1.8)]
    axes[0].legend(handles, ['Within-year oscillation', 'Same day each year'], loc='lower right',
                   ncol=2, fontsize=9)
    fig.tight_layout(h_pad=0.6)
    _save(fig, 'figS2')


# ── S3: realized germination ───────────────────────────────────────────────────
def _flux(args):
    germ, inf, years = args
    m = _model(germ, inf, years)
    sol = solve_ivp(m.df, m.t, m.X_0, dense_output=True, rtol=1e-6, atol=1e-9)
    day = np.arange(0, 365, 0.5)
    X = sol.sol(365 * (years - 1) + day)
    return day, germination(temp(day + FLOWERING_MIDPOINT - germ)) * X[6]


def fig_s3(years=300):
    with Pool(len(S3_MIDPOINTS)) as p:
        rows = p.map(_flux, [(g, _ess_for(g), years) for g in S3_MIDPOINTS])
    day = rows[0][0]
    _rc()
    fig, axes = plt.subplots(3, 1, figsize=(7, 5.2), sharex=True)
    for ax, g, (_, f) in zip(axes, S3_MIDPOINTS, rows):
        f = f / f.max()
        ax.axvspan(*FLOWER_ON, color=GRAY, alpha=0.3, lw=0)
        ax.fill_between(day, f, color=PINK, alpha=0.3, lw=0)
        ax.plot(day, f, color=PINK, lw=1.8)
        ax.axvline(g, color='#111827', ls=':', lw=1.2)
        ax.text(g + 3, 1.02, f'midpoint {g:g}', fontsize=9, va='top',
                bbox=dict(fc='white', ec='none', pad=1, alpha=0.85))
        ax.set_ylim(0, 1.15)
        ax.set_yticks([])
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
    axes[0].text(np.mean(FLOWER_ON), 1.17, 'Healthy flowering', ha='center', va='bottom',
                 color='#6b7280', fontsize=9)
    axes[1].text(85, 0.5, 'spring\ncohort', ha='left', color=PINK, fontsize=9)
    axes[1].text(238, 0.42, 'summer\ncohort', ha='left', color=PINK, fontsize=9)
    axes[-1].set_xlim(0, 365)
    axes[-1].set_xticks(np.arange(0, 366, 60))
    axes[-1].set_xlabel('Day of year')
    axes[1].set_ylabel('Seeds germinating (relative)')
    fig.tight_layout(h_pad=0.8)
    _save(fig, 'figS3')


if __name__ == '__main__':
    fig_s1()
    fig_s2()
    fig_s3()
