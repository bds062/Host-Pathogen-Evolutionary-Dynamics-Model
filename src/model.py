# model.py
import math
import numpy as np
from scipy.integrate import solve_ivp, simpson

from timing import (
    floweringS, vegetatingS, floweringI, vegetatingI,
    germination, temp
)

class Model:
    def __init__(self, **kwargs):
        self.X_0 = (5, 0, 0, 2.5, 0, 0, 0, 2.5, 0, 0)
        self.t = (0, 365 * 300)

        self.Bj = 0.2 / 365
        self.Bv = 0.025 / 365
        self.Bf = 0.025 / 365
        self.births = 4 / 365
        self.gamma = 0.0001
        self.death = .5 / 365
        self.maturity = 8 / 365
        self.g = 2 / 365

        self.susceptible_time = 182.5
        self.infected_time = 182.5
        self.infected_time2 = 182.5
        self.germination_time = 182.5

        self.eval = 6

        for key, value in kwargs.items():
            setattr(self, key, value)

    def run_sim(self):
        self.solution = solve_ivp(self.df, self.t, self.X_0)

        self.Sj = self.solution.y[0, :]
        self.Sv = self.solution.y[1, :]
        self.Sf = self.solution.y[2, :]
        self.Ij = self.solution.y[3, :]
        self.Iv = self.solution.y[4, :]
        self.If = self.solution.y[5, :]
        self.Sd = self.solution.y[6, :]
        self.Ij2 = self.solution.y[7, :]
        self.Iv2 = self.solution.y[8, :]
        self.If2 = self.solution.y[9, :]

        self.time_points = self.solution.t

        return self.solution

    def df(self, t, X):
        Sj, Sv, Sf, Ij, Iv, If, Sd, Ij2, Iv2, If2 = X

        susceptible_offset = 182.5 - self.susceptible_time
        infected_offset = 182.5 - self.infected_time
        infected_offset2 = 182.5 - self.infected_time2
        germination_offset = 182.5 - self.germination_time

        tCS = temp(t + susceptible_offset)
        tCI = temp(t + infected_offset)
        tCI2 = temp(t + infected_offset2)
        tCG = temp(t + germination_offset)

        dSd = self.births * Sf - germination(tCG) * Sd - (self.death / 4) * Sd
        dSj = germination(tCG) * Sd - self.Bj * Sj * (If + If2) - self.gamma * Sj * (
                Sv + Sf + Iv + If + Iv2 + If2) - self.maturity * Sj - self.death * Sj
        dSv = self.maturity * Sj - self.Bv * Sv * (If + If2) - floweringS(tCS) * Sv + vegetatingS(
            tCS) * Sf - self.death * Sv
        dSf = floweringS(tCS) * Sv - self.Bf * Sf * (If + If2) - vegetatingS(tCS) * Sf - self.death * Sf
        dIj = self.Bj * Sj * If - self.maturity * Ij - self.death * Ij
        dIv = self.maturity * Ij + self.Bv * Sv * If - floweringI(tCI) * Iv + vegetatingI(
            tCI) * If - self.death * Iv
        dIf = floweringI(tCI) * Iv + self.Bf * Sf * If - vegetatingI(tCI) * If - self.death * If
        dIj2 = self.Bj * Sj * If2 - self.maturity * Ij2 - self.death * Ij2
        dIv2 = self.maturity * Ij2 + self.Bv * Sv * If2 - floweringI(tCI2) * Iv2 + vegetatingI(
            tCI2) * If2 - self.death * Iv2
        dIf2 = floweringI(tCI2) * Iv2 + self.Bf * Sf * If2 - vegetatingI(tCI2) * If2 - self.death * If2

        return (dSj, dSv, dSf, dIj, dIv, dIf, dSd, dIj2, dIv2, dIf2)

    def evaluate(self, solution):
        class1_rows = [4, 5, 3]
        class2_rows = [7, 8, 9]

        class1 = solution.y[class1_rows, :].sum(axis=0)
        class2 = solution.y[class2_rows, :].sum(axis=0)

        max1, min1 = -math.inf, math.inf
        max2, min2 = -math.inf, math.inf

        for temp_t in range(int(len(class1) * 0.8), int(len(class1))):
            if class1[temp_t] > max1:
                max1 = class1[temp_t]
            elif class1[temp_t] < min1:
                min1 = class1[temp_t]
            if class2[temp_t] > max2:
                max2 = class2[temp_t]
            elif class2[temp_t] < min2:
                min2 = class2[temp_t]

        match self.eval:
            case 0:
                return simpson(class1, x=solution.t)
            case 1:
                return simpson(class2, x=solution.t)
            case 2:
                return (
                    simpson(class1, x=solution.t)
                    / (
                        simpson(class1, x=solution.t)
                        + simpson(class2, x=solution.t)
                    )
                )
            case 3:
                return (max1 + min1) / 2
            case 4:
                return (max2 + min2) / 2
            case 5:
                return ((max1 + min1) / 2) - ((max2 + min2) / 2)
            case 6:
                return ((max1 + min1) / 2) / (((max1 + min1) / 2) + ((max2 + min2) / 2))
            case _:
                return 0
