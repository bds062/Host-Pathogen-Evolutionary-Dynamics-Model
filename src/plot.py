"""Equilibrium-prevalence sweeps and the prevalence figures (Figures 2 and 3).

A sweep runs the single-genotype model to its annual cycle for each value of one timing
parameter and records the equilibrium prevalence (Model.evaluate). Results are cached as
.npz "embeddings" so a figure can be redrawn without repeating the simulations.
"""
import json
import os
import time
from multiprocessing import Pool

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from model import Model
from parameters import FLOWERING_MIDPOINT
from timing import floweringS, germination, temp

RC = {
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 13, 'axes.linewidth': 1.2,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'xtick.direction': 'out',
    'ytick.direction': 'out', 'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.major.size': 4, 'ytick.major.size': 4, 'legend.fontsize': 11,
    'legend.frameon': False, 'lines.linewidth': 2.0, 'pdf.fonttype': 42, 'ps.fonttype': 42,
}
GERMINATION_PINK = '#e75480'


# ── simulation and caching ──────────────────────────────────────────────────────
def _run_one(args):
    index, param_name, value, model_kwargs = args
    model = Model(**model_kwargs)
    setattr(model, param_name, value)
    return index, model.evaluate(model.run_sim())


def sweep(param_name, values, model_kwargs, num_processes):
    """Equilibrium prevalence for each `value` of `param_name`, other parameters fixed."""
    values = np.asarray(values, dtype=float)
    out = np.zeros(len(values))
    start = time.time()
    with Pool(processes=num_processes) as pool:
        for i, (index, result) in enumerate(pool.imap(
                _run_one, [(i, param_name, v, model_kwargs) for i, v in enumerate(values)])):
            out[index] = result
            if (i + 1) % max(len(values) // 4, 1) == 0 or i + 1 == len(values):
                print(f'  {i + 1}/{len(values)} done, {time.time() - start:.0f} s')
    return out


def save_embedding(path, x_name, x_values, curves, labels, metadata=None):
    """Write sweep curves to .npz (atomically) so the figure can be redrawn later."""
    meta = {'x_name': x_name, 'labels': list(labels)}
    meta.update(metadata or {})
    tmp = os.fspath(path) + '.tmp.npz'
    np.savez(tmp, x_values=np.asarray(x_values, float), curves=np.asarray(curves, float),
             labels=np.array(list(labels)), metadata=np.array(json.dumps(meta)))
    os.replace(tmp, os.fspath(path))
    print(f'Embedding saved to: {path}')


def load_embedding(path, x_values, labels):
    """Cached curves if `path` holds a sweep over exactly these x values and labels."""
    if path is None or not os.path.exists(path):
        return None
    with np.load(os.fspath(path), allow_pickle=False) as data:
        if [str(l) for l in data['labels']] != list(labels):
            return None
        if len(data['x_values']) != len(x_values) or not np.allclose(data['x_values'], x_values):
            return None
        return [row for row in np.asarray(data['curves'])]


def peaks(x_values, curves, labels):
    """Location and height of each curve's maximum."""
    x_values = np.asarray(x_values, float)
    return {label: (float(x_values[int(np.nanargmax(c))]), float(np.nanmax(c)))
            for label, c in zip(labels, curves)}


# ── figures ─────────────────────────────────────────────────────────────────────
def _shade(ax, x_fine, intensity, color, max_alpha):
    """Vertical shading whose opacity follows a seasonal rate."""
    alpha = intensity / intensity.max() * max_alpha if intensity.max() > 0 else intensity
    for i in range(len(x_fine) - 1):
        if alpha[i] > 1e-3:
            ax.axvspan(x_fine[i], x_fine[i + 1], color=color, alpha=alpha[i], lw=0, zorder=1)


def _style(ax, x_values, ylim, col):
    ax.set_xlim(x_values[0], x_values[-1])
    ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(plt.MultipleLocator(60))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(20))
    ax.tick_params(which='minor', length=2.5, width=0.8)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.7, alpha=0.5, zorder=0)
    if col == 0:
        ax.set_ylabel('Prevalence', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _ylim(all_results):
    vals = np.concatenate([c for panel in all_results for c in panel])
    return (0, max(np.ceil(np.nanmax(vals) * 20) / 20, 0.05))


def _annotate(ax, label, x, y, color, fontsize, dx):
    ax.annotate(label, xy=(x, y), xytext=(x + dx, y), fontsize=fontsize, color=color,
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))


def prevalence_vs_germination(base_kwargs, germination_values, maturity_values, bj_values,
                              maturity_labels, bj_labels, colors, save_path, embedding_path,
                              num_processes, figsize=(13, 4), arrow_fontsize=11):
    """Figure 2: equilibrium prevalence against germination midpoint. One panel per
    juvenile transmission coefficient (bj_values), one line per maturation rate."""
    germination_values = np.asarray(germination_values, float)
    labels = [f'{b} | {m}' for b in bj_labels for m in maturity_labels]
    cached = load_embedding(embedding_path, germination_values, labels)
    if cached is not None:
        print(f'Reusing {embedding_path}')
        n = len(maturity_values)
        results = [cached[i * n:(i + 1) * n] for i in range(len(bj_values))]
    else:
        results = []
        for bj in bj_values:
            panel = []
            for m in maturity_values:
                print(f'Bj = {bj:.5f}, m = {m:.5f}')
                panel.append(sweep('germination_time', germination_values,
                                   dict(base_kwargs, Bj=bj, maturity=m), num_processes))
            results.append(panel)
        save_embedding(embedding_path, 'germination_time', germination_values,
                       [c for p in results for c in p], labels,
                       {'bj_values': list(map(float, bj_values)),
                        'maturity_values': list(map(float, maturity_values))})
    for label, (day, value) in peaks(germination_values, [c for p in results for c in p],
                                     labels).items():
        print(f'  {label:40s} peak at germination midpoint {day:6.1f}: prevalence {value:.3f}')

    x_fine = np.linspace(germination_values[0], germination_values[-1], 500)
    flowering = floweringS(temp(x_fine))
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, len(bj_values), figsize=figsize, sharey=True, squeeze=False)
    axes = axes[0]
    ylim = _ylim(results)
    for col, (ax, panel, bj_label, letter) in enumerate(zip(axes, results, bj_labels, 'ABCD')):
        _shade(ax, x_fine, flowering, 'dimgray', 0.3)
        ax.axvline(FLOWERING_MIDPOINT, color='dimgray', linestyle=':', linewidth=1.4, zorder=2)
        for curve, label, color in zip(panel, maturity_labels, colors):
            ax.plot(germination_values, curve, label=label, color=color, zorder=3)
        ax.text(0.04, 1.05, letter, transform=ax.transAxes, fontsize=14, fontweight='bold',
                va='top', ha='left')
        ax.text(0.5, 1.04, bj_label, transform=ax.transAxes, fontsize=16, ha='center',
                va='bottom')
        if col == 0:
            _annotate(ax, 'Flowering season', FLOWERING_MIDPOINT, ylim[1] * 0.92, 'dimgray',
                      arrow_fontsize, 60)
        _style(ax, germination_values, ylim, col)
    fig.supxlabel('Germination midpoint (day)', fontsize=14, y=0.02)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles + [mpatches.Patch(color='dimgray', alpha=0.2)], names + ['Flowering'],
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.10), frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Figure saved to: {save_path}')
    plt.close(fig)
    return results


def prevalence_vs_flowering(base_kwargs, flowering_values, maturity_values,
                            germination_values, maturity_labels, germination_labels, colors,
                            save_path, embedding_path, num_processes, figsize=(12, 4.6),
                            arrow_fontsize=11):
    """Figure 3: equilibrium prevalence against the infectious flowering midpoint. One
    panel per germination midpoint (germination_values), one line per maturation rate."""
    flowering_values = np.asarray(flowering_values, float)
    labels = [f'{g} | {m}' for g in germination_labels for m in maturity_labels]
    cached = load_embedding(embedding_path, flowering_values, labels)
    if cached is not None:
        print(f'Reusing {embedding_path}')
        n = len(maturity_values)
        results = [cached[i * n:(i + 1) * n] for i in range(len(germination_values))]
    else:
        results = []
        for g in germination_values:
            panel = []
            for m in maturity_values:
                print(f'germination midpoint {g}, m = {m:.5f}')
                kw = dict(base_kwargs, germination_time=g, maturity=m)
                curve = sweep('infected_time', flowering_values, kw, num_processes)
                panel.append(curve)
            results.append(panel)
        save_embedding(embedding_path, 'infected_time', flowering_values,
                       [c for p in results for c in p], labels,
                       {'germination_values': list(map(float, germination_values)),
                        'maturity_values': list(map(float, maturity_values))})
    for label, (day, value) in peaks(flowering_values, [c for p in results for c in p],
                                     labels).items():
        print(f'  {label:40s} peak at flowering midpoint {day:6.1f}: prevalence {value:.3f}')

    x_fine = np.linspace(flowering_values[0], flowering_values[-1], 500)
    flowering = floweringS(temp(x_fine))
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, len(germination_values), figsize=figsize, sharey=True)
    ylim = _ylim(results)
    for col, (ax, panel, g, g_label, letter) in enumerate(
            zip(axes, results, germination_values, germination_labels, 'ABC')):
        _shade(ax, x_fine, flowering, 'dimgray', 0.45)
        _shade(ax, x_fine, germination(temp(x_fine + FLOWERING_MIDPOINT - g)), GERMINATION_PINK,
               0.45)
        for curve, label, color in zip(panel, maturity_labels, colors):
            ax.plot(flowering_values, curve, label=label, color=color, zorder=3)
        ax.axvline(g, color='dimgray', linewidth=1.3, linestyle='--', zorder=4)
        ax.axvline(FLOWERING_MIDPOINT, color='dimgray', linewidth=1.3, linestyle='--', zorder=4)
        if col == 0:
            dx = (flowering_values[-1] - flowering_values[0]) * 0.15 + FLOWERING_MIDPOINT - g
            _annotate(ax, 'Germination season', g, ylim[1] * 0.92, GERMINATION_PINK,
                      arrow_fontsize, dx)
            _annotate(ax, 'Flowering season', FLOWERING_MIDPOINT, ylim[1] * 0.84, 'dimgray',
                      arrow_fontsize, (flowering_values[-1] - flowering_values[0]) * 0.15)
        ax.text(0.04, 1.05, letter, transform=ax.transAxes, fontsize=14, fontweight='bold',
                va='top', ha='left')
        ax.text(0.5, 1.04, g_label, transform=ax.transAxes, fontsize=13, ha='center',
                va='bottom')
        _style(ax, flowering_values, ylim, col)
    fig.supxlabel('Infectious flowering midpoint (day)', fontsize=14, y=0.02)
    handles = [plt.Line2D([0], [0], color=c) for c in colors]
    fig.legend(handles + [mpatches.Patch(color=GERMINATION_PINK, alpha=0.45),
                          mpatches.Patch(color='dimgray', alpha=0.45)],
               list(maturity_labels) + ['Germination', 'Flowering'], loc='lower center',
               ncol=5, bbox_to_anchor=(0.5, -0.10), frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Figure saved to: {save_path}')
    plt.close(fig)
    return results
