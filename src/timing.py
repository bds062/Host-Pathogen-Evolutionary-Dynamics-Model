# helpers.py
import multiprocessing as mp
import numpy as np
import math

from scipy.integrate import solve_ivp, simpson
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

cores = mp.cpu_count()

TEMP_THRESHOLD=25
B_MAX=.5
STEEPNESS=.5

def floweringS(temp, steepness=STEEPNESS, threshold=TEMP_THRESHOLD, b_max=B_MAX):
    return b_max / (1 + np.exp(-steepness * (temp - threshold)))

def vegetatingS(temp, steepness=STEEPNESS, threshold=TEMP_THRESHOLD, b_max=B_MAX):
    return b_max - b_max / (1 + np.exp(-steepness * (temp - threshold)))

def floweringI(temp, steepness=STEEPNESS, threshold=TEMP_THRESHOLD, b_max=B_MAX):
    return b_max / (1 + np.exp(-steepness * (temp - threshold)))

def vegetatingI(temp, steepness=STEEPNESS, threshold=TEMP_THRESHOLD, b_max=B_MAX):
    return b_max - b_max / (1 + np.exp(-steepness * (temp - threshold)))

def germination(temp, steepness=STEEPNESS, threshold=TEMP_THRESHOLD, b_max=B_MAX, b_min=0):
    return b_max / (1 + np.exp(-steepness * (temp - threshold))) + b_min

def temp(t, min=0, scale=30):
    t = t * (2 * np.pi / 365) - np.pi
    temp_val = ((np.cos(t) + 1) / 2)
    temp_val = temp_val * scale
    temp_val = temp_val + min
    return temp_val

def tempVector(t):
    temps = np.zeros(len(t))
    for i, t_point in enumerate(t):
        temps[i] = temp(t_point)
    return temps
