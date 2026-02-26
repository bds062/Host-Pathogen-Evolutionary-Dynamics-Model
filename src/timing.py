# helpers.py
import multiprocessing as mp
import numpy as np
import math

from scipy.integrate import solve_ivp, simpson
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

cores = mp.cpu_count()

def floweringS(temp, steepness=5, threshold=25, b_max=2):
    return b_max / (1 + np.exp(-steepness * (temp - threshold)))

def vegetatingS(temp, steepness=5, threshold=25, b_max=2):
    return b_max - b_max / (1 + np.exp(-steepness * (temp - threshold)))

def floweringI(temp, steepness=5, threshold=25, b_max=2):
    return b_max / (1 + np.exp(-steepness * (temp - threshold)))

def vegetatingI(temp, steepness=5, threshold=25, b_max=2):
    return b_max - b_max / (1 + np.exp(-steepness * (temp - threshold)))

def germination(temp, steepness=5, threshold=30, b_max=1, b_min=0):
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
