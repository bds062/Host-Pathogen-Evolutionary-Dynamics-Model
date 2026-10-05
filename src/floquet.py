# floquet.py
"""
Invasion fitness via Floquet exponents (figs 3.4, 4.4, 5.2).

While genotype 2 is rare, its infected classes (Ij2, Iv2, If2) obey a LINEAR
system whose coefficients depend only on the resident's annual cycle
(Sj, Sv, Sf, total N) and genotype 2's own flowering schedule:

    d/dt [Ij2, Iv2, If2] = A(t) [Ij2, Iv2, If2],   A(t + 365) = A(t)

Integrating dPhi/dt = A(t) Phi, Phi(0) = I over one year gives the monodromy
matrix M = Phi(365). Its dominant eigenvalue rho is the mutant's exact
year-over-year multiplication factor once start-up transients have died out,
so r = ln(rho) (per year) is the invasion fitness s(resident, mutant): the
long-run growth rate a rare mutant would show in an infinitely long run.
r(x, x) = 0 when the resident sits on its periodic orbit (used as a check).

ESS search per germination time:
  * Selection gradient D(x) = d s(x, y)/dy at y = x (central difference, +/- delta days).
  * Singular strategies are sign changes of D around the (periodic) flowering-time
    circle. D: + -> - is convergence stable (evolution moves toward it).
  * Each convergence-stable point is refined by root-finding on D between grid
    points, then checked against a dense (1-day) set of mutants:
      ESS         : no mutant anywhere can invade  (max s <= tol)
      branching   : convergence stable but invadable by nearby mutants (d2s/dy2 > 0)
      local ESS   : locally uninvadable but some distant mutant invades
"""
import os
import json
import time
import multiprocessing as mp

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from model import Model
from timing import floweringI, vegetatingI, temp

PERIOD = 365.0
PARAM_NAMES = ('Bj', 'Bv', 'Bf', 'births', 'gamma', 'death', 'maturity', 'g',
               'susceptible_time', 'germination_time')
KIND_CODES = {'ESS': 0, 'branching': 1, 'local ESS': 2, 'repeller': 3}
KIND_NAMES = {v: k for k, v in KIND_CODES.items()}

EQ_RTOL, EQ_ATOL = 1e-9, 1e-12       # resident equilibration
FL_RTOL, FL_ATOL = 1e-10, 1e-13      # one-year Floquet integration


def safe_label(label):
    return (label.replace(' ', '_').replace('.', '').replace('(', '')
            .replace(')', '').replace('×', 'x'))


def model_kwargs_from(model, **overrides):
    kw = {name: getattr(model, name) for name in PARAM_NAMES}
    kw.update(overrides)
    return kw


def _make_model(model_kwargs, resident_time):
    kw = dict(model_kwargs)
    kw['infected_time'] = float(resident_time) % PERIOD
    kw['infected_time2'] = float(resident_time) % PERIOD
    return Model(**kw)


def equilibrate_resident(model, x0=None, min_years=400, max_years=5000,
                         chunk_years=100, tol=1e-8):
    """Run the resident alone until its state repeats year to year.

    Returns (state at a year boundary, years run, relative one-year residual)."""
    X = np.array(model.X_0 if x0 is None else x0, dtype=float)
    X[7:10] = 0.0

    def advance(state, n_years):
        sol = solve_ivp(model.df, (0.0, PERIOD * n_years), state,
                        method='DOP853', rtol=EQ_RTOL, atol=EQ_ATOL)
        return sol.y[:, -1]

    years = 0
    if min_years > 0:
        X = advance(X, min_years)
        years += min_years
    while True:
        X_next = advance(X, 1)
        years += 1
        resid = np.max(np.abs(X_next - X)) / max(np.max(np.abs(X_next)), 1e-12)
        X = X_next
        if resid < tol or years >= max_years:
            return X, years, float(resid)
        X = advance(X, chunk_years - 1)
        years += chunk_years - 1


def floquet_row(model, X_res, mutant_times):
    """Invasion exponents r (per year) of rare mutants with flowering times
    `mutant_times` against the resident whose year-boundary state is X_res."""
    mutant_times = np.asarray(mutant_times, dtype=float)
    K = len(mutant_times)
    offsets = 182.5 - mutant_times
    p = model
    X10 = np.zeros(10)

    def rhs(t, Z):
        X10[:7] = Z[:7]
        dX = np.asarray(p.df(t, X10), dtype=float)[:7]
        Sj, Sv, Sf, Ij, Iv, If, Sd = Z[:7]
        N = Sj + Sv + Sf + Ij + Iv + If + Sd
        tc = temp(t + offsets)
        fI = floweringI(tc)
        vI = vegetatingI(tc)
        A = np.zeros((K, 3, 3))
        A[:, 0, 0] = -p.gamma * N - p.maturity - p.death
        A[:, 0, 2] = p.Bj * Sj
        A[:, 1, 0] = p.maturity
        A[:, 1, 1] = -fI - p.death
        A[:, 1, 2] = p.Bv * Sv + vI
        A[:, 2, 1] = fI
        A[:, 2, 2] = p.Bf * Sf - vI - p.death
        dPhi = A @ Z[7:].reshape(K, 3, 3)
        return np.concatenate([dX, dPhi.ravel()])

    Z0 = np.concatenate([np.asarray(X_res, dtype=float)[:7],
                         np.tile(np.eye(3), (K, 1, 1)).ravel()])
    sol = solve_ivp(rhs, (0.0, PERIOD), Z0, method='DOP853', rtol=FL_RTOL, atol=FL_ATOL)
    M = sol.y[7:, -1].reshape(K, 3, 3)
    rho = np.array([np.max(np.abs(np.linalg.eigvals(Mk))) for Mk in M])
    with np.errstate(divide='ignore'):
        return np.log(rho)


# ── Phase 1: grid rows ─────────────────────────────────────────────────────────
def _grid_task(args):
    key, model_kwargs, resident_time, mutant_times, delta = args
    model = _make_model(model_kwargs, resident_time)
    X, years, resid = equilibrate_resident(model)
    muts = np.concatenate([mutant_times, [resident_time - delta, resident_time,
                                          resident_time + delta]]) % PERIOD
    r = floquet_row(model, X, muts)
    infected = X[3] + X[4] + X[5]
    total = X[:7].sum()
    return key, {
        'row': r[:-3], 'r_minus': r[-3], 'r_self': r[-2], 'r_plus': r[-1],
        'state': X, 'years': years, 'resid': resid,
        'infected_frac': infected / total if total > 0 else 0.0,
    }


# ── Phase 2: refine convergence-stable singular strategies ─────────────────────
def _gradient_at(model_kwargs, x, delta, warm_state):
    model = _make_model(model_kwargs, x)
    X, _, _ = equilibrate_resident(model, x0=warm_state, min_years=0)
    r = floquet_row(model, X, np.array([x - delta, x, x + delta]) % PERIOD)
    return (r[2] - r[0]) / (2 * delta), X


def _refine_task(args):
    key, model_kwargs, xa, xb, warm_state, delta, x_guess = args
    state = {'warm': warm_state}

    def D(x):
        d, X = _gradient_at(model_kwargs, x, delta, state['warm'])
        state['warm'] = X
        return d

    try:
        x_star = brentq(D, xa, xb, xtol=0.02, maxiter=40)
    except ValueError:
        x_star = x_guess  # grid bracket did not reproduce; keep interpolated estimate

    model = _make_model(model_kwargs, x_star)
    X, years, resid = equilibrate_resident(model, x0=state['warm'], min_years=0)
    dense = np.arange(0.0, PERIOD, 1.0)
    muts = np.concatenate([dense, [x_star - delta, x_star, x_star + delta]]) % PERIOD
    r = floquet_row(model, X, muts)
    r_minus, r_self, r_plus = r[-3], r[-2], r[-1]
    D_star = (r_plus - r_minus) / (2 * delta)
    C_star = (r_plus + r_minus - 2 * r_self) / delta ** 2
    max_invasion = float(np.max(r[:-3]))
    tol = max(1e-8, 10 * abs(r_self))
    if C_star > 0:
        kind = 'branching'
    elif max_invasion <= tol:
        kind = 'ESS'
    else:
        kind = 'local ESS'
    return key, {
        'x': float(x_star % PERIOD), 'kind': kind, 'D': float(D_star), 'C': float(C_star),
        'max_invasion': max_invasion, 'argmax_invader': float(dense[np.argmax(r[:-3])]),
        'r_self': float(r_self), 'resid': resid,
        'dense_row': r[:-3],
    }


def _singular_brackets(xs, D, viable):
    """Sign changes of D around the circle of unique grid points."""
    n = len(xs)
    out = []
    for a in range(n):
        b = (a + 1) % n
        if not (viable[a] and viable[b] and np.isfinite(D[a]) and np.isfinite(D[b])):
            continue
        xa, xb = xs[a], xs[b] + (PERIOD if b == 0 else 0.0)
        if D[a] > 0 and D[b] <= 0:
            kind = 'attractor'
        elif D[a] < 0 and D[b] >= 0:
            kind = 'repeller'
        else:
            continue
        guess = xa + (xb - xa) * D[a] / (D[a] - D[b])
        out.append((a, xa, xb, guess, kind))
    return out


def compute_floquet_sweep(configs, germination_values, flowering_values, num_processes,
                          save_path, delta=0.5, max_singular=6):
    """configs: dict label -> model_kwargs. Writes everything to `save_path` (npz)."""
    germ = np.asarray(germination_values, dtype=float)
    flow = np.asarray(flowering_values, dtype=float)
    labels = list(configs)
    nG, nF = len(germ), len(flow)
    duplicate_endpoint = np.isclose(flow[-1] - flow[0], PERIOD)
    n_unique = nF - 1 if duplicate_endpoint else nF

    tasks = []
    for label in labels:
        for g_idx, g in enumerate(germ):
            kw = dict(configs[label], germination_time=float(g))
            for i in range(n_unique):
                tasks.append(((label, g_idx, i), kw, float(flow[i]), flow, delta))

    grid = {label: {
        'rows': np.full((nG, nF, nF), np.nan), 'r_minus': np.full((nG, nF), np.nan),
        'r_self': np.full((nG, nF), np.nan), 'r_plus': np.full((nG, nF), np.nan),
        'resid': np.full((nG, nF), np.nan), 'years': np.zeros((nG, nF)),
        'infected_frac': np.full((nG, nF), np.nan), 'states': np.full((nG, nF, 10), np.nan),
    } for label in labels}

    start = time.time()
    print(f"Phase 1: {len(tasks)} resident rows on {num_processes} processes", flush=True)
    with mp.Pool(num_processes) as pool:
        for done, (key, res) in enumerate(pool.imap_unordered(_grid_task, tasks, chunksize=1), 1):
            label, g_idx, i = key
            G = grid[label]
            G['rows'][g_idx, i] = res['row']
            for name in ('r_minus', 'r_self', 'r_plus', 'resid', 'years', 'infected_frac'):
                G[name][g_idx, i] = res[name]
            G['states'][g_idx, i] = res['state']
            if done % 100 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} rows  ({time.time() - start:.0f}s)", flush=True)

    if duplicate_endpoint:  # day 365 == day 0
        for G in grid.values():
            for name in ('r_minus', 'r_self', 'r_plus', 'resid', 'years', 'infected_frac'):
                G[name][:, -1] = G[name][:, 0]
            G['rows'][:, -1, :] = G['rows'][:, 0, :]
            G['states'][:, -1] = G['states'][:, 0]

    # Gradient, curvature, viability, singular brackets
    xs = flow[:n_unique]
    refine_tasks = []
    brackets = {}
    for label in labels:
        G = grid[label]
        G['D'] = (G['r_plus'] - G['r_minus']) / (2 * delta)
        G['C'] = (G['r_plus'] + G['r_minus'] - 2 * G['r_self']) / delta ** 2
        G['viable'] = G['infected_frac'] > 1e-6
        for g_idx in range(nG):
            br = _singular_brackets(xs, G['D'][g_idx, :n_unique], G['viable'][g_idx, :n_unique])
            brackets[(label, g_idx)] = br
            kw = dict(configs[label], germination_time=float(germ[g_idx]))
            for s_idx, (a, xa, xb, guess, kind) in enumerate(br):
                if kind == 'attractor':
                    refine_tasks.append(((label, g_idx, s_idx), kw, xa, xb,
                                         G['states'][g_idx, a], delta, guess))

    print(f"Phase 2: refining {len(refine_tasks)} convergence-stable points", flush=True)
    refined = {}
    with mp.Pool(num_processes) as pool:
        for key, res in pool.imap_unordered(_refine_task, refine_tasks, chunksize=1):
            refined[key] = res
            print(f"  {key[0]} germ={germ[key[1]]:.1f}: x*={res['x']:.2f} {res['kind']} "
                  f"(C={res['C']:.2e}, max invasion={res['max_invasion']:.2e} "
                  f"by y={res['argmax_invader']:.0f})", flush=True)

    arrays = {'germination_values': germ, 'flowering_values': flow,
              'labels': np.array(labels), 'delta': delta,
              'metadata': json.dumps({'configs': configs, 'eq_rtol': EQ_RTOL,
                                      'fl_rtol': FL_RTOL}, default=float)}
    out = {}
    for label in labels:
        G = grid[label]
        s = safe_label(label)
        sing_x = np.full((nG, max_singular), np.nan)
        sing_kind = np.full((nG, max_singular), -1)
        sing_C = np.full((nG, max_singular), np.nan)
        sing_maxinv = np.full((nG, max_singular), np.nan)
        dense_rows = np.full((nG, max_singular, int(PERIOD)), np.nan)
        for g_idx in range(nG):
            for s_idx, (a, xa, xb, guess, kind) in enumerate(brackets[(label, g_idx)][:max_singular]):
                if kind == 'attractor':
                    res = refined[(label, g_idx, s_idx)]
                    sing_x[g_idx, s_idx] = res['x']
                    sing_kind[g_idx, s_idx] = KIND_CODES[res['kind']]
                    sing_C[g_idx, s_idx] = res['C']
                    sing_maxinv[g_idx, s_idx] = res['max_invasion']
                    dense_rows[g_idx, s_idx] = res['dense_row']
                else:
                    sing_x[g_idx, s_idx] = guess % PERIOD
                    sing_kind[g_idx, s_idx] = KIND_CODES['repeller']
        ess = _primary_ess(sing_x, sing_kind)
        out[label] = {'ess': ess, 'sing_x': sing_x, 'sing_kind': sing_kind}
        arrays.update({
            f'{s}_invasion': G['rows'], f'{s}_D': G['D'], f'{s}_C': G['C'],
            f'{s}_r_self': G['r_self'], f'{s}_viable': G['viable'],
            f'{s}_infected_frac': G['infected_frac'], f'{s}_resid': G['resid'],
            f'{s}_years': G['years'], f'{s}_ess': ess,
            f'{s}_sing_x': sing_x, f'{s}_sing_kind': sing_kind, f'{s}_sing_C': sing_C,
            f'{s}_sing_max_invasion': sing_maxinv, f'{s}_sing_dense_row': dense_rows,
        })
    tmp = os.fspath(save_path) + '.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, os.fspath(save_path))
    print(f"Saved {save_path}  (total {time.time() - start:.0f}s)", flush=True)
    return out


def _primary_ess(sing_x, sing_kind):
    """One ESS per germination time; if several, follow the previous point."""
    nG = sing_x.shape[0]
    ess = np.full(nG, np.nan)
    prev = np.nan
    for g in range(nG):
        cands = sing_x[g][sing_kind[g] == KIND_CODES['ESS']]
        if len(cands) == 0:
            continue
        if len(cands) == 1 or not np.isfinite(prev):
            pick = cands[0]
        else:
            dist = np.abs((cands - prev + PERIOD / 2) % PERIOD - PERIOD / 2)
            pick = cands[np.argmin(dist)]
        ess[g] = pick
        prev = pick
    return ess


# ── Plotting ───────────────────────────────────────────────────────────────────
def _rc():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 11, 'axes.linewidth': 1.2,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
        'legend.frameon': False, 'lines.linewidth': 2.0, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def coexistence_partners(S, flow, x):
    """Flowering times that form a protected dimorphism with `x`: each invades the
    other's monoculture when rare (s > 0 both ways). `S` is the invasion matrix for one
    germination time, S[resident, mutant]; `x` is snapped to the nearest grid row."""
    i = int(np.argmin(np.abs(flow - x)))
    with np.errstate(invalid='ignore'):
        mask = (S[i, :] > 0) & (S[:, i] > 0)
    mask[i] = False
    return np.flatnonzero(mask)


def css_track(germ, sing_x, sing_kind):
    """The convergence-stable flowering time at each germination time, chained left to
    right for continuity. Unlike the global ESS this always exists here, so the curve
    has no gaps; `is_global` flags the points that no mutant at all can invade.

    Returns (track, is_global), both length len(germ), NaN where nothing converged.
    """
    stable = (KIND_CODES['ESS'], KIND_CODES['local ESS'], KIND_CODES['branching'])
    track = np.full(len(germ), np.nan)
    is_global = np.zeros(len(germ), dtype=bool)
    prev = None
    for g in range(len(germ)):
        cands = [(float(sing_x[g, k]), sing_kind[g, k] == KIND_CODES['ESS'])
                 for k in range(sing_kind.shape[1]) if sing_kind[g, k] in stable]
        if not cands:
            continue
        glob = [c for c in cands if c[1]]
        if glob:
            pick = glob[0]
        elif prev is None:
            pick = cands[0]
        else:  # nearest on the circle, so the branch does not jump spuriously
            pick = min(cands, key=lambda c: abs((c[0] - prev + PERIOD / 2) % PERIOD - PERIOD / 2))
        track[g], is_global[g] = pick
        prev = track[g]
    return track, is_global


def plot_floquet_ess(embedding_path, labels=None, legend_fmt='{}', colors=None,
                     line_alpha=1.0, figsize=(7, 6), title=None, save_path=None, dpi=300,
                     label_map=None, connect_css=False, show_coexistence=False):
    """ESS flowering time vs germination time, one line per label.
    Filled markers/line: ESS. Open circles: branching points. Crosses: local-only ESS.

    label_map:   {stored label: label to display}, for relabelling a legend without
                 touching the embedding (the stored labels stay the array keys).
    connect_css: draw the convergence-stable branch instead of the global ESS, so the
                 curve is unbroken where no global ESS exists (crosses mark those points).
    show_coexistence: at those germination times, also draw the flowering times that
                 form a protected dimorphism with the convergence-stable strategy.
    """
    with np.load(os.fspath(embedding_path), allow_pickle=False) as data:
        germ = np.asarray(data['germination_values'])
        if labels is None:
            labels = [str(l) for l in data['labels']]
        keys = ('ess', 'sing_x', 'sing_kind') + (('invasion',) if show_coexistence else ())
        flow = np.asarray(data['flowering_values'])
        loaded = {l: {k: np.asarray(data[f'{safe_label(l)}_{k}']) for k in keys}
                  for l in labels}
    for label, d in loaded.items():
        d['css'], d['css_global'] = css_track(germ, d['sing_x'], d['sing_kind'])
    if colors is None:
        colors = ['#2563eb', '#dc2626', '#16a34a', '#f59e0b', '#7c3aed']

    _rc()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([germ[0], germ[-1]], [germ[0], germ[-1]], color='#888888', linewidth=1.2,
            linestyle='--', zorder=1, label='ESS = germination time')
    dodge_step = 3.0  # nudge each config's dimorphism bar so overlapping ones stay visible
    for k, ((label, d), color) in enumerate(zip(loaded.items(), colors)):
        y = d['css'] if connect_css else d['ess']
        shown = (label_map or {}).get(label, label)
        ax.plot(germ, y, color=color, alpha=line_alpha, linewidth=2.0, zorder=3,
                label=legend_fmt.format(shown))
        on_line = d['css_global'] if connect_css else np.isfinite(d['ess'])
        ax.plot(germ[on_line], y[on_line], color=color, alpha=line_alpha, linestyle='none',
                marker='o', markersize=5, markeredgewidth=0.8, markeredgecolor='white',
                zorder=3)
        kinds = [(KIND_CODES['branching'], 'o')]  # noqa: E501
        if not show_coexistence:  # otherwise the dimorphism bar marks these instead
            kinds.append((KIND_CODES['local ESS'], 'x'))
        for code, marker in kinds:
            g_idx, s_idx = np.nonzero(d['sing_kind'] == code)
            if len(g_idx):
                kw = ({'facecolors': 'none', 'edgecolors': color} if marker == 'o'
                      else {'color': color})
                ax.scatter(germ[g_idx], d['sing_x'][g_idx, s_idx], s=40, marker=marker,
                           alpha=line_alpha, linewidths=1.2, zorder=4, **kw)
        if show_coexistence:
            # No global ESS and a coexisting partner are the same fact: the endpoint is
            # a pair. Open circle = the convergence-stable morph, bar = the flowering
            # times it coexists with (the second morph).
            for g in np.flatnonzero(np.isfinite(d['css']) & ~d['css_global']):
                partners = coexistence_partners(d['invasion'][g], flow, d['css'][g])
                if not len(partners):
                    continue
                x = germ[g] + (k - (len(loaded) - 1) / 2) * dodge_step
                for run in np.split(partners, np.flatnonzero(np.diff(partners) > 1) + 1):
                    ax.plot([x, x], [flow[run[0]], flow[run[-1]]], color=color,
                            alpha=0.45 * line_alpha, linewidth=7, solid_capstyle='butt',
                            zorder=2)
                ax.plot([germ[g]], [d['css'][g]], linestyle='none', marker='o', markersize=6,
                        markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.6,
                        alpha=line_alpha, zorder=5)
        # extra ESSs not on the primary line
        g_idx, s_idx = np.nonzero(d['sing_kind'] == KIND_CODES['ESS'])
        extra = ~np.isclose(d['sing_x'][g_idx, s_idx], d['ess'][g_idx])
        if np.any(extra):
            ax.scatter(germ[g_idx[extra]], d['sing_x'][g_idx[extra], s_idx[extra]], s=30,
                       color=color, alpha=line_alpha, zorder=4)

    handles, texts = ax.get_legend_handles_labels()
    entries = [('o', 'none', 'branching point')]
    if not show_coexistence:
        entries.append(('x', 'k', 'no globally uninvadable strategy'))
    for marker, face, text in entries:
        if any(np.any(d['sing_kind'] == KIND_CODES[{'o': 'branching', 'x': 'local ESS'}[marker]])
               for d in loaded.values()):
            handles.append(plt.Line2D([], [], linestyle='none', marker=marker, color='k',
                                      markerfacecolor=face, markersize=6))
            texts.append(text)
    if show_coexistence:
        handles.append(plt.Line2D([], [], linestyle='none', marker='o', color='k',
                                  markerfacecolor='white', markeredgewidth=1.6, markersize=7))
        texts.append('dimorphism: first morph (no single ESS here)')
        handles.append(plt.Line2D([], [], color='k', alpha=0.45, linewidth=7))
        texts.append('dimorphism: flowering times it coexists with')
    ax.set_xlim(germ[0] - 10, germ[-1] + 10)
    ax.set_ylim(0, PERIOD)
    ax.xaxis.set_major_locator(plt.MultipleLocator(60))
    ax.yaxis.set_major_locator(plt.MultipleLocator(60))
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Germination Time (day of year)')
    ax.set_ylabel(('Convergence-Stable' if connect_css else 'ESS')
                  + ' Flowering Time (day of year)')
    ax.set_title(title or 'Evolutionarily Stable Infectious Flowering Time\nvs Germination Timing'
                          '\n(invasion fitness, Floquet exponent)',
                 fontsize=12, fontweight='bold', pad=10)
    ax.legend(handles, texts, loc='upper left', fontsize=9, frameon=False)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    plt.close(fig)
    return loaded


def outcome_matrix(S):
    """Which genotype wins each pair, as one signed number per cell.

    S[i, j] is s(resident i, mutant j): the growth rate of the column genotype while
    rare against the row genotype. With a = S and b = S.T:

        a > 0 > b : only the column genotype can invade -> it wins   (positive)
        b > 0 > a : only the row genotype can invade    -> it wins   (negative)
        a > 0, b > 0 : each invades the other -> coexistence         (exactly 0)

    Away from coexistence the magnitude is (a - b)/2, the mean of the two consistent
    signals. The result is exactly antisymmetric (out.T == -out), so the raster mirrors
    across the diagonal, and the neutral colour marks the polymorphic pairs.
    """
    a, b = S, S.T
    with np.errstate(invalid='ignore'):
        return np.where((a > 0) & (b > 0), 0.0, (a - b) / 2)


def _raster_panel(ax, values, flow, sing_x, sing_kind, cmap, norm, contour=None):
    """One invasion raster on the resident x mutant plane (figs 5.2 and 6).

    Draws the grid itself, the neutral diagonal and the singular strategies; the caller
    supplies the colour mapping (continuous invasion fitness, or the mutual-invasibility
    categories) and adds its own colourbar and title.
    """
    step = flow[1] - flow[0]
    extent = (flow[0] - step / 2, flow[-1] + step / 2, flow[0] - step / 2, flow[-1] + step / 2)
    im = ax.imshow(values, origin='lower', extent=extent, cmap=cmap, norm=norm,
                   aspect='equal', interpolation='nearest')
    if contour is not None:
        ax.contour(flow, flow, contour, levels=[0.0], colors='k', linewidths=1.0)
    ax.axline((0, 0), (1, 1), color='k', linewidth=0.6, alpha=0.5)
    for x, k in zip(sing_x, sing_kind):
        if k == KIND_CODES['ESS']:
            ax.axhline(x, color='k', linestyle='--', linewidth=1.5)
            ax.axvline(x, color='k', linestyle='--', linewidth=1.5)
            ax.text(x + 4, 6, f'ESS = {x:.1f}', fontsize=10, fontweight='bold')
        elif k in (KIND_CODES['branching'], KIND_CODES['local ESS']):
            ax.axhline(x, color='#555555', linestyle=':', linewidth=1.2)
            ax.axvline(x, color='#555555', linestyle=':', linewidth=1.2)
            ax.text(x + 4, 20, f'{KIND_NAMES[int(k)]} = {x:.1f}', fontsize=9)
    ax.set_xlim(0, PERIOD)
    ax.set_ylim(0, PERIOD)
    ax.set_xlabel('Mutant (invader) flowering time (day of year)')
    ax.set_ylabel('Resident flowering time (day of year)')
    return im


def _load_raster(embedding_path, label, germination_time):
    """Invasion matrix + singular strategies for one config at one germination time,
    with non-viable residents blanked out."""
    s = safe_label(label)
    with np.load(os.fspath(embedding_path), allow_pickle=False) as data:
        germ = np.asarray(data['germination_values'])
        flow = np.asarray(data['flowering_values'])
        g_idx = int(np.argmin(np.abs(germ - germination_time)))
        S = np.asarray(data[f'{s}_invasion'])[g_idx].copy()
        S[~np.asarray(data[f'{s}_viable'])[g_idx], :] = np.nan
        return (S, flow, float(germ[g_idx]),
                np.asarray(data[f'{s}_sing_x'])[g_idx],
                np.asarray(data[f'{s}_sing_kind'])[g_idx])


def plot_floquet_pip(embedding_path, label, germination_time, linthresh=1e-4,
                     figsize=None, save_path=None, dpi=300, display_label=None,
                     outcome=False):
    """Pairwise invasibility plot: invasion fitness s(resident y-axis, mutant x-axis).

    `germination_time` may be a list, giving one panel per germination day on a shared
    colour scale. outcome=True plots outcome_matrix() instead: the same red/blue
    winner colouring, but mirrored across the diagonal, with coexisting (polymorphic)
    pairs left at the neutral midpoint of the scale.
    """
    times = np.atleast_1d(germination_time)
    panels = [_load_raster(embedding_path, label, t) for t in times]
    if outcome:
        panels = [(outcome_matrix(S), *rest) for S, *rest in panels]
    vmax = max(np.nanmax(np.abs(S)) for S, *_ in panels)
    norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#e5e5e5')

    _rc()
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=figsize or ((8, 7) if n == 1 else (6.6 * n, 6.2)),
                             squeeze=False)
    for ax, (S, flow, g, sing_x, sing_kind) in zip(axes[0], panels):
        im = _raster_panel(ax, S, flow, sing_x, sing_kind, cmap, norm,
                           contour=None if outcome else S)
        if outcome:  # no resident/mutant roles left once the plot is symmetric
            ax.set_xlabel('Genotype 1 flowering time (day of year)')
            ax.set_ylabel('Genotype 2 flowering time (day of year)')
        if n > 1:
            ax.set_title(f'Germination day {g:.1f}', fontsize=11, fontweight='bold')
    name = 'Competitive outcome' if outcome else 'Pairwise invasibility plot'
    subtitle = ('red = genotype 1 wins, blue = genotype 2 wins, white = they coexist'
                if outcome else 'invasion fitness (per year; red = invader grows)')
    if n == 1:
        axes[0, 0].set_title(f'{name} — {display_label or label}, '
                             f'germination day {panels[0][2]:.1f}\n{subtitle}',
                             fontsize=12, fontweight='bold', pad=10)
    else:
        fig.suptitle(f'{name} — {display_label or label}\n{subtitle}',
                     fontsize=12, fontweight='bold')
    cbar = fig.colorbar(im, ax=axes[0, 0] if n == 1 else axes[0].tolist(), shrink=0.8)
    cbar.set_label(('Winner and margin; 0 = coexistence' if outcome
                    else 'Invasion fitness s') + ' (yr$^{-1}$, symlog)')
    if n == 1:
        fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    plt.close(fig)
    return [S for S, *_ in panels] if n > 1 else (panels[0][0], panels[0][3], panels[0][4])
