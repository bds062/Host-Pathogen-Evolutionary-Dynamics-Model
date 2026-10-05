# Host–Pathogen Evolutionary Dynamics Model

A seasonally forced model of *Silene latifolia* and anther smut, used to find the
evolutionarily stable **infectious flowering time** and how it depends on host
**germination timing**.

$$\begin{align*}
\dot{S_j} &= b(t)S_f - S_j(\beta_j I_f + \gamma N + m + \mu)\\
\dot{S_v} &= m S_j + v(t)_s S_f - S_v(\beta_v  I_f + f(t)_s + \mu)\\
\dot{S_f} &= f(t)_s S_v - S_f(\beta_f I_f + v(t)_s + \mu)\\
\dot{I_j} &= \beta_j S_j I_f - I_j(m + \mu)\\
\dot{I_v} &= m I_j + \beta_v S_v I_f  + v(t)_I I_f - I_v(f(t)_I + \mu)\\
\dot{I_f} &= f(t)_I I_v + I_f(\beta_f S_f - v(t)_I - \mu)
\end{align*}$$

<img width="852" alt="model schematic" src="https://github.com/user-attachments/assets/fd5890a1-25bb-449b-b6a3-91de70949038">

---

## Competition rasters across germination times

![ESS rasters animation](docs/figures/ess_rasters.gif)

## Prevalence

Prevalence peaks when germination precedes flowering by ~60 days. Slow maturation
(4 yr⁻¹) drives the pathogen extinct at mistimed flowering.

![prevalence vs germination](docs/figures/fig1.6.png)
![prevalence vs flowering time](docs/figures/fig2.6.Dianthus.png)

## Evolutionarily stable flowering time

The ESS tracks germination with a lag and jumps between an early and a late branch near
germination day 120.

| maturation (× default 8 yr⁻¹) | species parameters |
|---|---|
| ![fig3.4](docs/figures/fig3.4.png) | ![fig4.4](docs/figures/fig4.4.png) |

Along the convergence-stable branch, with dimorphisms marked:

| | |
|---|---|
| ![fig3.5](docs/figures/fig3.5.png) | ![fig4.5](docs/figures/fig4.5.png) |

## Pairwise invasibility

Red = the rare mutant grows. The ESS is the resident whose row is entirely blue.

![PIP](docs/figures/fig5.2.png)

## Polymorphism

At germination day 119 some pairs are red both ways — each invades the other — so no
flowering time is uninvadable. Confirmed by simulating the full model from both
directions (`figs/dimorphism_checks.txt`).

![PIP at both germination times](docs/figures/fig6.png)

Same data by outcome: red = genotype 1 wins, blue = genotype 2 wins, white = coexistence.

![competitive outcome](docs/figures/fig7.png)

---

## Running

```bash
pip install -r requirements.txt
cd test
python figs.py --list            # figure keys
python figs.py --fig 6 7         # redraw from cached embeddings (seconds)
python figs.py --fig 3_4         # full invasion-fitness sweep (~1 h on 32 cores)
```

On a cluster: `SB_ACCOUNT=myacct ./scripts/submit.sh 3_4`

`src/` model, seasonal forcing, plotting, invasion fitness (`floquet.py`) ·
`test/` figure and confirmation scripts · `figs/` outputs
