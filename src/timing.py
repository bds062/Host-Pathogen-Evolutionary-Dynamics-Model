"""Seasonal forcing.

Temperature follows a cosine with a one-year period (0 at the start of the year, SCALE at
mid-year). Each seasonal rate is a logistic switch on temperature around THRESHOLD, so it
is near its maximum B_MAX for roughly 98 days per year, centred on the day its phase-
shifted temperature peaks. Shifting the temperature argument by (182.5 - midpoint) moves
that window to any midpoint of the year.
"""
import numpy as np

THRESHOLD = 25          # temperature at which a seasonal switch is half on
B_MAX = 0.5             # peak value of every seasonal rate (per day)
STEEPNESS = 0.5         # logistic steepness of the switch
SCALE = 30              # annual temperature amplitude


def temp(t, min=0, scale=SCALE):
    """Temperature on day t (0 at day 0, `scale` at day 182.5)."""
    t = t * (2 * np.pi / 365) - np.pi
    return (np.cos(t) + 1) / 2 * scale + min


def _switch_on(temp, steepness=STEEPNESS, threshold=THRESHOLD, b_max=B_MAX):
    return b_max / (1 + np.exp(-steepness * (temp - threshold)))


def germination(temp, steepness=STEEPNESS, threshold=THRESHOLD, b_max=B_MAX, b_min=0):
    """Seed germination rate g(t)."""
    return _switch_on(temp, steepness, threshold, b_max) + b_min


def floweringS(temp, steepness=STEEPNESS, threshold=THRESHOLD, b_max=B_MAX):
    """Vegetative -> flowering transition of healthy plants, f_S(t)."""
    return _switch_on(temp, steepness, threshold, b_max)


def vegetatingS(temp, steepness=STEEPNESS, threshold=THRESHOLD, b_max=B_MAX):
    """Flowering -> vegetative transition of healthy plants, v_S(t)."""
    return b_max - _switch_on(temp, steepness, threshold, b_max)


def floweringI(temp, steepness=STEEPNESS, threshold=THRESHOLD, b_max=B_MAX):
    """Vegetative -> flowering transition of infected plants, f_I(t)."""
    return _switch_on(temp, steepness, threshold, b_max)


def vegetatingI(temp, steepness=STEEPNESS, threshold=THRESHOLD, b_max=B_MAX):
    """Flowering -> vegetative transition of infected plants, v_I(t)."""
    return b_max - _switch_on(temp, steepness, threshold, b_max)
