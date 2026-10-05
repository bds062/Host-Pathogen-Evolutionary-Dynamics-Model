"""
Confirm by direct simulation that the dimorphisms drawn in figs 3.5/4.5 are real.

The Floquet sweeps (floquet.py, figs 3.4/4.4) flag a germination time as having no
globally uninvadable strategy whenever the convergence-stable flowering time x* can be
invaded by a distant mutant y that can also be invaded back:

    s(x*, y) > 0  and  s(y, x*) > 0

That is the standard protected-coexistence criterion, but it is a linearised,
rare-invader statement. This script runs the full two-genotype model instead: it
equilibrates one morph alone, seeds the other at `--seed`, integrates for `--years`,
and does the same with the roles reversed. A genuine protected dimorphism converges to
the SAME interior equilibrium from both directions; that agreement is the pass test.

Two facts about the timescale matter here. A rare invader near a singular strategy
grows at ~1e-3/yr, so climbing from 1e-6 to a visible share takes thousands of years —
400 years is not enough and will read as "resident wins" in both directions (this is
exactly the artefact that the old share-after-400-years ESS metric suffered from).
Checkpoints are therefore reported at several horizons, so a run that has not yet
converged is visible rather than silently wrong.

Usage:
    python confirm_dimorphism.py                       # every flagged case, both embeddings
    python confirm_dimorphism.py --embedding fig3.4    # just the maturity sweep
    python confirm_dimorphism.py --config Basic --germination 119
    python confirm_dimorphism.py --pair 93.6 187.2 --config Basic --germination 119
    python confirm_dimorphism.py --years 1600 --out ../figs/dimorphism_checks.txt
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
REPO_DIR = TEST_DIR.parent
FIGS_DIR = REPO_DIR / 'figs'
sys.path.append(str(REPO_DIR / 'src'))

import multiprocessing as mp

import numpy as np
from scipy.integrate import solve_ivp

from model import Model
from floquet import (PERIOD, coexistence_partners, css_track, equilibrate_resident,
                     safe_label)

EMBEDDINGS = {'fig3.4': FIGS_DIR / 'fig3.4_embedding.npz',
              'fig4.4': FIGS_DIR / 'fig4.4_embedding.npz'}
DEFAULT_YEARS = (400, 1600, 3200)
SHARE_TOL = 1e-3          # how closely the two directions must agree to pass

try:
    NUM_PROCESSES = len(os.sched_getaffinity(0))
except AttributeError:
    NUM_PROCESSES = os.cpu_count()


def flagged_cases(embedding_path, config=None, germination=None, partner=None):
    """Every (config, germination, x*, partner) the sweep flagged as dimorphic.

    The partner is the flowering time with the strongest mutual invasion, i.e. the
    largest min(s(x*, y), s(y, x*)) — the most robustly protected pair on the grid,
    and so the least likely to be a boundary artefact.
    """
    with np.load(os.fspath(embedding_path), allow_pickle=False) as data:
        germ = np.asarray(data['germination_values'])
        flow = np.asarray(data['flowering_values'])
        labels = [str(l) for l in data['labels']]
        meta = json.loads(str(data['metadata'])) if 'metadata' in data else {}
        cases = []
        for label in labels:
            if config is not None and label != config:
                continue
            s = safe_label(label)
            S_all = np.asarray(data[f'{s}_invasion'])
            css, is_global = css_track(germ, np.asarray(data[f'{s}_sing_x']),
                                       np.asarray(data[f'{s}_sing_kind']))
            for g in np.flatnonzero(np.isfinite(css) & ~is_global):
                if germination is not None and abs(germ[g] - germination) > 1e-6:
                    continue
                if partner is not None:
                    cases.append((label, float(germ[g]), float(css[g]), float(partner),
                                  meta.get('configs', {}).get(label, {})))
                    continue
                idx = coexistence_partners(S_all[g], flow, css[g])
                if not len(idx):
                    continue
                i = int(np.argmin(np.abs(flow - css[g])))
                strength = np.minimum(S_all[g][i, idx], S_all[g][idx, i])
                cases.append((label, float(germ[g]), float(css[g]),
                              float(flow[idx[int(np.argmax(strength))]]),
                              meta.get('configs', {}).get(label, {})))
        return cases


def _direction(args):
    """One direction: `resident` alone to its cycle, then `invader` seeded rare."""
    key, model_kwargs, resident, invader, years, seed = args
    kw = dict(model_kwargs)
    kw.update(infected_time=resident, infected_time2=invader)
    model = Model(**kw)
    X, eq_years, resid = equilibrate_resident(model)
    X = np.asarray(X, dtype=float).copy()
    X[9] = seed                                   # seed the invader as If2
    out = []
    for T in years:
        sol = solve_ivp(model.df, (0.0, PERIOD * T), X, method='DOP853',
                        rtol=1e-9, atol=1e-12)
        y = sol.y[:, -1]
        res_mass, inv_mass = float(y[3:6].sum()), float(y[7:10].sum())
        total = res_mass + inv_mass
        out.append({'years': T, 'resident_mass': res_mass, 'invader_mass': inv_mass,
                    'invader_share': inv_mass / total if total > 0 else float('nan')})
    return key, {'resident': resident, 'invader': invader, 'eq_years': eq_years,
                 'eq_residual': resid, 'checkpoints': out}


def confirm(cases, years=DEFAULT_YEARS, seed=1e-6, num_processes=None):
    """Run both directions of every case (in parallel across directions)."""
    tasks = []
    for n, (label, germ, x_star, partner, cfg) in enumerate(cases):
        kw = dict(cfg, germination_time=germ)
        kw.pop('infected_time', None)
        kw.pop('infected_time2', None)
        tasks.append(((n, 0), kw, x_star, partner, years, seed))
        tasks.append(((n, 1), kw, partner, x_star, years, seed))

    results = {}
    start = time.time()
    with mp.Pool(min(num_processes or NUM_PROCESSES, len(tasks))) as pool:
        for done, (key, res) in enumerate(pool.imap_unordered(_direction, tasks), 1):
            results[key] = res
            print(f"  {done}/{len(tasks)} directions done ({time.time() - start:.0f}s)",
                  flush=True)
    return results


def report(cases, results, years, tol=SHARE_TOL):
    """Human-readable report; returns (text, all_passed)."""
    lines, passed = [], True
    for n, (label, germ, x_star, partner, _) in enumerate(cases):
        fwd, rev = results[(n, 0)], results[(n, 1)]
        lines.append(f"\n{label} — germination day {germ:.1f}")
        lines.append(f"  convergence-stable morph {x_star:.1f} vs partner {partner:.1f}")
        for tag, r in (("seeded partner", fwd), ("seeded morph  ", rev)):
            marks = "  ".join(f"yr {c['years']}: {c['invader_share']:.4f}"
                              for c in r['checkpoints'])
            lines.append(f"    {tag} (resident {r['resident']:6.1f}): {marks}")
        # Both directions must end on the same interior equilibrium: the share of the
        # SAME morph, so one of the two is flipped before comparing.
        f_end, r_end = fwd['checkpoints'][-1], rev['checkpoints'][-1]
        share_partner_fwd = f_end['invader_share']
        share_partner_rev = 1.0 - r_end['invader_share']
        interior = min(share_partner_fwd, share_partner_rev) > tol and \
            max(share_partner_fwd, share_partner_rev) < 1 - tol
        agree = abs(share_partner_fwd - share_partner_rev) <= tol
        ok = interior and agree
        passed &= ok
        lines.append(f"    partner share at yr {years[-1]}: {share_partner_fwd:.4f} "
                     f"(from morph) vs {share_partner_rev:.4f} (from partner)  "
                     f"-> {'DIMORPHISM CONFIRMED' if ok else 'NOT CONFIRMED'}"
                     + ("" if interior else "; one morph was lost")
                     + ("" if agree else "; directions disagree, try more --years"))
    return "\n".join(lines), passed


def main():
    parser = argparse.ArgumentParser(
        description="Simulate the dimorphisms flagged in figs 3.5/4.5 to confirm they are real.")
    parser.add_argument('--embedding', nargs='+', choices=sorted(EMBEDDINGS),
                        default=sorted(EMBEDDINGS), help="Which sweep(s) to read cases from.")
    parser.add_argument('--config', help="Only this config label (e.g. Basic, '2×').")
    parser.add_argument('--germination', type=float, help="Only this germination day.")
    parser.add_argument('--pair', nargs=2, type=float, metavar=('X', 'Y'),
                        help="Force this flowering-time pair instead of the sweep's choice.")
    parser.add_argument('--years', nargs='+', type=int, default=list(DEFAULT_YEARS),
                        help=f"Checkpoint horizons in years (default {list(DEFAULT_YEARS)}).")
    parser.add_argument('--seed', type=float, default=1e-6, help="Rare-invader seed (If2).")
    parser.add_argument('--processes', type=int, default=NUM_PROCESSES)
    parser.add_argument('--out', help="Also write the report to this path.")
    args = parser.parse_args()

    cases = []
    for name in args.embedding:
        path = EMBEDDINGS[name]
        if not path.exists():
            print(f"{path} not found — run figs.py --fig {name.replace('fig', '').replace('.', '_')}"
                  " first.")
            continue
        found = flagged_cases(path, args.config, args.germination,
                              args.pair[1] if args.pair else None)
        if args.pair:  # --pair also overrides the first morph
            found = [(l, g, args.pair[0], args.pair[1], cfg) for l, g, _, _, cfg in found]
        cases.extend(found)

    if not cases:
        print("No dimorphic cases matched — every flagged strategy is a global ESS here.")
        return 0

    print(f"{len(cases)} case(s), {len(args.years)} checkpoints, "
          f"up to {max(args.years)} years each:")
    for label, germ, x_star, partner, _ in cases:
        print(f"  {label}, germination {germ:.1f}: {x_star:.1f} vs {partner:.1f}")

    results = confirm(cases, tuple(args.years), args.seed, args.processes)
    text, passed = report(cases, results, tuple(args.years))
    print(text)
    print(f"\n{'All cases confirmed.' if passed else 'Some cases NOT confirmed (see above).'}")
    if args.out:
        Path(args.out).write_text(
            f"Dimorphism confirmation — seed {args.seed:g}, horizons {args.years}\n" + text + "\n")
        print(f"Report written to: {args.out}")
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
