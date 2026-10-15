"""Host parameter sets used in the manuscript (rates per day).

SILENE   Silene latifolia (Slowinski et al. 2025; Bruns et al. 2022): the main parameter
         set, Table 1 of the manuscript.
DIANTHUS Dianthus pavonius (Bruns et al. 2017): the comparison host, Figure 6.

Keys match the attributes of model.Model:
  Bj, Bv, Bf   transmission coefficients for juvenile, vegetative and flowering plants
  births       seeds produced per healthy flowering plant
  gamma        density-dependent mortality coefficient (juveniles only)
  death        background mortality (seeds die at death / 4)
  maturity     juvenile -> adult maturation rate
  g            seed germination scale (the seasonal rate itself is set in timing.py)
"""

SILENE = dict(Bj=0.6 / 365, Bv=0.4 / 365, Bf=0.4 / 365, births=8 / 365, gamma=1e-3,
              death=0.36 / 365, maturity=0.85 / 365, g=2 / 365)

DIANTHUS = dict(Bj=3.38e-4, Bv=3.38e-5, Bf=3.38e-5, births=1.3 / 365, gamma=1.5e-4,
                death=0.187 / 365, maturity=8 / 365, g=2 / 365)

SPECIES = {'Silene latifolia': SILENE, 'Dianthus pavonius': DIANTHUS}

# Timing conventions shared by every analysis: healthy flowering is centred on mid-year,
# and the standard initial condition is 5 susceptible and 5 infected juveniles.
FLOWERING_MIDPOINT = 182.5
X0 = (5, 0, 0, 5, 0, 0, 0, 0, 0, 0)
