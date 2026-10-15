"""When do seeds actually germinate?

`germination_time` is the midpoint of the germination window (the day the per-seed rate
peaks). Because the rate is already substantial weeks before that day, over-wintered seeds
germinate early and the realized timing depends on when seeds are produced. For each
germination midpoint of the species sweep (figs/sweep_species.npz) this runs the single-
genotype model to its annual cycle, with infectious flowering at the convergence-stable
strategy, and reports the median germination day, the share of germination before day 130
(the spring cohort), the day juvenile abundance peaks, and the ESS offset from the median
germination day (the "44-60 days" of the Results).

    python realized_germination.py [--years 300] [--processes 16]   -> figs/realized_germination.txt
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / 'figs'
sys.path.insert(0, str(REPO / 'src'))

from floquet import css_track                      # noqa: E402
from model import Model                            # noqa: E402
from parameters import FLOWERING_MIDPOINT, SPECIES, X0  # noqa: E402
from timing import germination, temp               # noqa: E402


def realized(args):
    name, germ, inf, years = args
    m = Model(X_0=X0, t=(0, 365 * years), germination_time=germ, infected_time=inf,
              infected_time2=inf, **SPECIES[name])
    sol = solve_ivp(m.df, m.t, m.X_0, dense_output=True, rtol=1e-6, atol=1e-9)
    day = np.arange(0, 365, 0.5)
    X = sol.sol(365 * (years - 1) + day)
    flux = germination(temp(day + FLOWERING_MIDPOINT - germ)) * X[6]   # seeds germinating per day
    order = np.argsort((day - (germ - FLOWERING_MIDPOINT)) % 365)      # year starts opposite the midpoint
    cum = np.cumsum(flux[order]) / flux.sum()
    median = day[order][np.searchsorted(cum, 0.5)]
    spring = flux[day < 130].sum() / flux.sum()
    juveniles = X[0] + X[3] + X[7]
    return name, germ, inf, median, spring, day[juveniles.argmax()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--embedding', default=str(FIGS / 'sweep_species.npz'))
    ap.add_argument('--years', type=int, default=300)
    ap.add_argument('--processes', type=int, default=16)
    ap.add_argument('--out', default=str(FIGS / 'realized_germination.txt'))
    a = ap.parse_args()

    d = np.load(a.embedding)
    germs = d['germination_values']
    jobs = []
    for name in SPECIES:
        lab = name.replace(' ', '_')
        css, _ = css_track(germs, d[f'{lab}_sing_x'], d[f'{lab}_sing_kind'])
        jobs += [(name, g, c, a.years) for g, c in zip(germs, css)]
    with Pool(a.processes) as pool:
        rows = pool.map(realized, jobs)

    lines = [f"{'config':18s} {'midpoint':>8s} {'median':>7s} {'spring':>6s} {'juv.peak':>8s} "
             f"{'ESS':>6s} {'ESS-median':>10s}"]
    for name, germ, inf, med, spring, jpk in rows:
        offset = (inf - med + FLOWERING_MIDPOINT) % 365 - FLOWERING_MIDPOINT
        lines.append(f"{name:18s} {germ:8.1f} {med % 365:7.1f} {spring:6.2f} {jpk:8.1f} "
                     f"{inf:6.1f} {offset:10.1f}")
    text = '\n'.join(lines)
    print(text)
    Path(a.out).write_text(text + '\n')
    print(f'written to {a.out}')


if __name__ == '__main__':
    main()
