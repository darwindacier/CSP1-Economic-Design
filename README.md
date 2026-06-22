# Economic Design and Optimization of CSP-1 Continuous Sampling Plans

Reproducibility package for the paper:

> **Economic Design and Optimization of CSP-1 Continuous Sampling Plans with a
> Four-Component Cost Model under Markovian Transition Dynamics**
> D. Peña-González, R. Torres-Peña, G. Solano-Navarro.
> Submitted to the *International Journal of Production Economics*.

This repository contains the model, optimizer, simulation, sensitivity/robustness
analyses, and the empirical validation on a real semiconductor stream (SECOM) used
in the paper. The manuscript itself is **not** included.

## Contents

| File | Description |
|------|-------------|
| `csp_model.py` | Four-component CSP-1 economic cost model (`CSP1EconomicModel`) and `optimize_csp1`, derived by renewal-reward analysis of a two-state Markov chain. |
| `optimize_economic.py` | Driver that optimizes the clearance number `i` (exhaustive search) and sampling fraction `f` (Brent's method). |
| `simulate_csp1.py` | Discrete-event simulation of the CSP-1 rule, used to validate the analytic model (agreement within 1%). |
| `validate_optimizer.py` | Cross-checks the optimizer against the simulation. |
| `verify_consistency.py` | Consistency checks (e.g. AFI vs. the classical CSP-1 characteristic). |
| `empirical_secom.py` | Estimates `p'` from the real SECOM stream and validates each policy by **replay** over the true defect sequence. |
| `run_sensitivity.py`, `run_sensitivity_extended.py` | Sensitivity of the optimum to the cost parameters and `p'`. |
| `run_robustness.py` | Robustness of incurred cost to misestimation of `p'`. |
| `generate_results_table.py`, `generate_heatmaps.py` | Reproduce the results tables and heatmaps. |
| `data/secom_labels.data` | Pass/fail labels of the SECOM data set, in temporal order (see Data sources). |
| `results/*.csv` | Pre-computed numerical outputs (regenerable by the scripts above). |

## Requirements

```
pip install -r requirements.txt
```

Python 3.10+ with `numpy`, `scipy`, `pandas`, `matplotlib`.

## Reproducing the results

Run from the repository root (scripts resolve paths relative to their own location):

```bash
python optimize_economic.py        # economic optimum
python simulate_csp1.py            # discrete-event validation
python verify_consistency.py       # AFI / model consistency checks
python run_sensitivity_extended.py # sensitivity study
python run_robustness.py           # robustness to p' misestimation
python empirical_secom.py          # SECOM estimation + replay validation
python generate_results_table.py   # results tables
python generate_heatmaps.py        # heatmaps
```

## Data sources

The SECOM data set is publicly available from the UCI Machine Learning Repository:

> McCann, M., Johnston, A. (2008). *SECOM Data Set*. UCI Machine Learning
> Repository. https://doi.org/10.24432/C54305

Only the pass/fail label file (`secom_labels.data`), in its original temporal order,
is redistributed here for convenience; it is used to estimate the empirical defect
rate and to replay the inspection policies over the real stream.

## License

The code in this repository is released under the MIT License (see `LICENSE`).
The SECOM data set retains its original UCI license and citation.
