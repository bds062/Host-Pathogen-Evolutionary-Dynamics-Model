"""Seasonally forced, age-structured host-pathogen model (Equations 4-10 of the manuscript).

State vector X = (Sj, Sv, Sf, Ij, Iv, If, Sd, Ij2, Iv2, If2):
  Sd            seeds
  Sj, Sv, Sf    susceptible juvenile, vegetative and flowering plants
  Ij, Iv, If    plants infected by pathogen genotype 1 (flowering midpoint `infected_time`)
  Ij2, Iv2, If2 plants infected by genotype 2 (midpoint `infected_time2`); zero unless a
                second genotype is seeded

Infected plants are sterile; only healthy flowering plants produce seeds. All transmission
is by contact with infectious flowering plants (If + If2). Juveniles additionally suffer
density-dependent mortality gamma * N.
"""
import numpy as np
from scipy.integrate import solve_ivp

from parameters import FLOWERING_MIDPOINT, SILENE, X0
from timing import floweringI, floweringS, germination, temp, vegetatingI, vegetatingS

PREVALENCE, GENOTYPE_1_SHARE = 7, 6     # `eval` modes of Model.evaluate


class Model:
    """Parameters default to the Silene latifolia set (parameters.SILENE). Seasonal
    timings are window midpoints (day of year): `germination_time`, `susceptible_time`
    (healthy flowering) and `infected_time` / `infected_time2` (infectious flowering)."""

    def __init__(self, **kwargs):
        self.X_0 = X0
        self.t = (0, 365 * 400)
        for key, value in SILENE.items():
            setattr(self, key, value)
        self.susceptible_time = FLOWERING_MIDPOINT
        self.infected_time = FLOWERING_MIDPOINT
        self.infected_time2 = FLOWERING_MIDPOINT
        self.germination_time = FLOWERING_MIDPOINT
        self.eval = PREVALENCE
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise TypeError(f'unknown Model parameter {key!r}')
            setattr(self, key, value)

    def run_sim(self):
        self.solution = solve_ivp(self.df, self.t, self.X_0)
        return self.solution

    def df(self, t, X):
        Sj, Sv, Sf, Ij, Iv, If, Sd, Ij2, Iv2, If2 = X
        tCS = temp(t + FLOWERING_MIDPOINT - self.susceptible_time)
        tCI = temp(t + FLOWERING_MIDPOINT - self.infected_time)
        tCI2 = temp(t + FLOWERING_MIDPOINT - self.infected_time2)
        tCG = temp(t + FLOWERING_MIDPOINT - self.germination_time)
        N = Sd + Sj + Sv + Sf + Ij + Iv + If + Ij2 + Iv2 + If2
        infectious = If + If2

        dSd = self.births * Sf - germination(tCG) * Sd - (self.death / 4) * Sd
        dSj = (germination(tCG) * Sd - self.Bj * Sj * infectious - self.gamma * Sj * N
               - self.maturity * Sj - self.death * Sj)
        dSv = (self.maturity * Sj - self.Bv * Sv * infectious - floweringS(tCS) * Sv
               + vegetatingS(tCS) * Sf - self.death * Sv)
        dSf = (floweringS(tCS) * Sv - self.Bf * Sf * infectious - vegetatingS(tCS) * Sf
               - self.death * Sf)

        dIj = self.Bj * Sj * If - self.gamma * Ij * N - self.maturity * Ij - self.death * Ij
        dIv = (self.maturity * Ij + self.Bv * Sv * If - floweringI(tCI) * Iv
               + vegetatingI(tCI) * If - self.death * Iv)
        dIf = floweringI(tCI) * Iv + self.Bf * Sf * If - vegetatingI(tCI) * If - self.death * If

        dIj2 = (self.Bj * Sj * If2 - self.gamma * Ij2 * N - self.maturity * Ij2
                - self.death * Ij2)
        dIv2 = (self.maturity * Ij2 + self.Bv * Sv * If2 - floweringI(tCI2) * Iv2
                + vegetatingI(tCI2) * If2 - self.death * Iv2)
        dIf2 = (floweringI(tCI2) * Iv2 + self.Bf * Sf * If2 - vegetatingI(tCI2) * If2
                - self.death * If2)
        return (dSj, dSv, dSf, dIj, dIv, dIf, dSd, dIj2, dIv2, dIf2)

    def evaluate(self, solution):
        """Summary statistic of a run over its final 20% (the periodic attractor).

        eval == PREVALENCE:       mean infected fraction of the whole population
        eval == GENOTYPE_1_SHARE: genotype 1's share of all infected plants, from the
                                  midpoint of each genotype's oscillation
        """
        g1 = solution.y[[3, 4, 5], :].sum(axis=0)
        g2 = solution.y[[7, 8, 9], :].sum(axis=0)
        window = slice(int(len(g1) * 0.8), len(g1))
        if self.eval == PREVALENCE:
            susceptible = solution.y[[0, 1, 2, 6], :].sum(axis=0)
            total = susceptible + g1 + g2
            return float(np.mean((g1[window] + g2[window]) / total[window]))
        if self.eval == GENOTYPE_1_SHARE:
            mid1 = (g1[window].max() + g1[window].min()) / 2
            mid2 = (g2[window].max() + g2[window].min()) / 2
            return float(mid1 / (mid1 + mid2))
        raise ValueError(f'unknown eval mode {self.eval}')
