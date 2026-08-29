# Leakage-Aware Explainable Ensemble Learning for Concrete Compressive Strength

Reproducibility package for:

> A. F. Alshaibanee, M. Habiban, K. A. M. Habeeban, and A. F. Mahdi, "A Leakage-Aware Explainable Ensemble Learning Framework for Concrete Compressive Strength Prediction Using SHAP-Based Engineering Interpretation", International Journal of Intelligent Engineering and Systems (under review, Paper ID 20265490).

## What this repository contains

```
code/
  concrete_xai_pipeline_R1_revised.py   # the single script that generates every number in the paper
results/
  environment_versions.json             # exact package versions and seeds used for the reported runs
  splits/
    kfold_split_indices.csv             # fixed 5-fold random-split indices (seed 42)
    groupkfold_split_indices.csv        # fixed formulation-grouped split indices
  tables/                               # CSV outputs backing every table of the paper
requirements.txt                        # pinned dependency versions
```

Mapping from paper tables to output files:

| Paper table | File |
|---|---|
| Table 2 (imputation comparison) | `tables/imputation_results.csv`, `tables/selected_imputers.json` |
| Tables 4-6 (Experiments 1-3, 5-fold CV) | `tables/main_experiments_cv.csv` (+ `main_experiments_fold_level.csv`) |
| Table 7 (group-wise validation) | `tables/groupwise_cv.csv`, `tables/logo_overall.csv` (+ per-group files) |
| Table 8 (Wilcoxon tests) | `tables/wilcoxon_tests.csv` |
| Section 4.4 redundancy figures | `tables/redundancy_analysis.json`, `tables/formulation_group_sizes.csv` |
| Section 4.4 perturbation test | `tables/perturbation_robustness.csv` |

## Data

The ConcreteXAI dataset (4,420 records, 11 variables) is publicly available; see:

> J. A. Guzman-Torres et al., "ConcreteXAI: A multivariate dataset for concrete strength prediction via deep-learning-based methods", Data in Brief, Vol. 53, Art. no. 110218, 2024. https://doi.org/10.1016/j.dib.2024.110218

Download the dataset, save it as `Data.csv` next to the script (or edit `DATA_PATH` at the top of the script). The expected columns are:
`Type_of_cement, Brand, Additives, Type_of_aggregates, Design_F'c (Mpa), Curing_age_(days), Cs_(Mpa), Ts_(Mpa), Fs_(Mpa), Er_(ohm-cm), UPV_(m/s)`.

## How to reproduce

```bash
pip install -r requirements.txt
python code/concrete_xai_pipeline_R1_revised.py
```

The script runs, in order: environment logging; formulation-group construction and redundancy analysis; leakage-aware imputation comparison (KNN vs Random Forest, predictor sets exclude all strength variables; selection criterion: minimum mean RMSE, tie-break by MAE); the three feature-configuration experiments under a fully nested 5-fold protocol (imputation, encoding, and scaling fitted inside each training fold only); group-wise validation (GroupKFold and leave-one-formulation-out); the measurement-perturbation test; and the Wilcoxon signed-rank tests. All outputs are written to `R1_RESULTS/`.

Everything is deterministic given `random_state = 42`. Small numeric differences may appear with other library versions than those pinned in `requirements.txt`; the paper reports values at the resolution (4 decimal places for R², 3 for errors) at which they are stable.

## License

Code released under the MIT License. The ConcreteXAI dataset retains its original license (CC BY 4.0); it is not redistributed here.
