# Leakage-Aware Explainable Ensemble Learning for Concrete Compressive Strength

Reproducibility package for:

> A. F. Alshaibanee, M. Habiban, K. A. M. Habeeban, and A. F. Mahdi, "A Leakage-Aware Explainable Ensemble Learning Framework for Concrete Compressive Strength Prediction Using SHAP-Based Engineering Interpretation", International Journal of Intelligent Engineering and Systems (under review, Paper ID 20265490).

## What this repository contains

```
code/
  concrete_xai_pipeline_R1_revised.py
data/
  Data.csv
results/
  environment_versions.json
  splits/
    kfold_split_indices.csv
    groupkfold_split_indices.csv
  tables/
requirements.txt
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

The ConcreteXAI dataset used in this study is included in the `data/` directory as `Data.csv`. The script automatically resolves the dataset path relative to the repository root, so no manual path configuration is required. The expected columns are:
`Type_of_cement, Brand, Additives, Type_of_aggregates, Design_F'c (Mpa), Curing_age_(days), Cs_(Mpa), Ts_(Mpa), Fs_(Mpa), Er_(ohm-cm), UPV_(m/s)`.

## How to reproduce

```bash
pip install -r requirements.txt
python code/concrete_xai_pipeline_R1_revised.py
```

The script runs, in order: environment logging; formulation-group construction and redundancy analysis; leakage-aware imputation comparison (KNN vs Random Forest, predictor sets exclude all strength variables; selection criterion: minimum mean RMSE, tie-break by MAE); the three feature-configuration experiments under a fully nested 5-fold protocol (imputation, encoding, and scaling fitted inside each training fold only); group-wise validation (GroupKFold and leave-one-formulation-out); the measurement-perturbation test; and the Wilcoxon signed-rank tests. All outputs are written to the `results/` directory.

Everything is deterministic given `random_state = 42`. Small numeric differences may appear with other library versions than those pinned in `requirements.txt`; the paper reports values at the resolution (4 decimal places for R², 3 for errors) at which they are stable.

## License

Code released under the MIT License. The ConcreteXAI dataset is redistributed under its original CC BY 4.0 license. Please cite the original dataset publication when using the dataset.


## Nested Optuna Optimization and Reproducibility (R1-7)

This repository provides reproducibility evidence for the leakage-aware prediction of concrete compressive strength using the ConcreteXAI dataset.

### Experimental Protocols

Two five-fold outer cross-validation protocols were evaluated:

- **Random5Fold:** Randomly shuffled five-fold cross-validation.
- **GroupKFold5:** Five-fold group-disjoint cross-validation designed to assess generalization to unseen concrete formulation groups.

Nested Optuna hyperparameter optimization was performed for the PolyBayes-ABR model, with 50 optimization trials per outer fold and 5 outer folds per protocol, yielding **500 recorded trials across both protocols**.

### Verified Results

The recorded outer-fold performance summaries are:

| Protocol | Mean R² | Mean RMSE (MPa) | Mean MAE (MPa) |
|---|---:|---:|---:|
| Random5Fold | 0.995129 | 0.668356 | 0.426619 |
| GroupKFold5 | 0.179571 | 5.322920 | 4.504512 |

These values are arithmetic means of the five outer-fold metrics for each protocol. They should not be interpreted as pooled out-of-fold metrics.

### Reproducibility Files

The updated repository includes:

- `code/concrete_xai_pipeline_R3_with_optuna_logging.py` — updated experiment pipeline and logging.
- `code/run_optuna_full_R1_7_FIXED.py` — nested optimization runner.
- `results/optuna_nested_R1_7/` — 26 CSV evidence files, including Optuna trial logs, inner-fold split records, best parameters, outer-fold metrics, and out-of-fold predictions.

The historical R1 code and earlier results are retained for provenance.

### Reproducibility Notes

The Optuna experiment completed successfully in the original execution environment. Its outputs were subsequently checked for file completeness and numerical consistency.

A separate clean-environment end-to-end reproduction has not been independently documented. The published results should therefore be distinguished from a newly reproduced run.

### Data and Software

The experiment uses the ConcreteXAI dataset. Consult the repository's dataset licensing information before redistributing or reusing the data.

Python package dependencies are documented in `requirements.txt`. Exact versions used in the original execution environment should be preserved separately when available.

### Scientific Interpretation

The difference between Random5Fold and GroupKFold5 performance demonstrates the importance of evaluating generalization beyond randomly partitioned observations. Strong random-fold accuracy alone does not establish reliable performance on unseen concrete formulation groups.
