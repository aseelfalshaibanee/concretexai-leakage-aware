# =========================================================
# ConcreteXAI Pipeline - R3 AUDITED (based on R2; reviewer-focused fixes)
# Addresses reviewer comments of the 1st review round:
#   R1.5  one internally consistent evaluation protocol for all tables
#   R1.6  independent refit of every model in Experiment 3
#   R1.8  explicit numerical criterion for imputation-method selection
#   R1.9  fixed split indices, seeds, versions exported for reproducibility
#   R2.4  group-wise validation (GroupKFold + leave-one-formulation-out)
#   R2.5  fully nested preprocessing/imputation (no test-fold information
#         reaches any fitted transformer or imputer)
#   R2.7  measurement-perturbation robustness test
#   R2.8  feature redundancy / effective-independence analysis
# =========================================================

import os
import sys
import json
import warnings
# Keep warnings visible during the audit; they may reveal compatibility or data issues.

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# =========================================================
# SETTINGS
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "Data.csv")
BASE_DIR = os.path.join(SCRIPT_DIR, "R3_AUDITED_RESULTS")
TARGET = "Cs_(Mpa)"
RANDOM_STATE = 42
N_SPLITS = 5
POLYBAYES_TRIALS = 50  # Use 2-5 only for a smoke test; restore 50 for final reported runs.

TABLES_DIR = os.path.join(BASE_DIR, "tables")
SPLITS_DIR = os.path.join(BASE_DIR, "splits")
for p in [BASE_DIR, TABLES_DIR, SPLITS_DIR]:
    os.makedirs(p, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("cat", encoder, cat_cols),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols)
    ])


# =========================================================
# ENVIRONMENT VERSIONS (R1.9)
# =========================================================

import sklearn
import xgboost
import lightgbm
import catboost
import scipy

versions = {
    "python": sys.version,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit-learn": sklearn.__version__,
    "xgboost": xgboost.__version__,
    "lightgbm": lightgbm.__version__,
    "catboost": catboost.__version__,
    "scipy": scipy.__version__,
    "random_state": RANDOM_STATE,
    "n_splits": N_SPLITS,
    "polybayes_trials_per_outer_fold": POLYBAYES_TRIALS,
}
with open(os.path.join(BASE_DIR, "environment_versions.json"), "w") as f:
    json.dump(versions, f, indent=2)
log(json.dumps(versions, indent=2))

# =========================================================
# LOAD DATA
# =========================================================

log("Loading data")
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
# Normalize leading/trailing whitespace in categorical values in memory only.
# The original Data.csv is never overwritten.
for _c in df.select_dtypes(include="object").columns:
    df[_c] = df[_c].astype("string").str.strip()
    df[_c] = df[_c].astype(object)
if df[TARGET].isna().any():
    raise ValueError(f"Target {TARGET} contains missing values; resolve before modeling.")
log(f"Shape: {df.shape}")
log(f"Columns: {list(df.columns)}")
log("Missing:\n" + df.isnull().sum().to_string())

CAT_COLS = df.select_dtypes(include="object").columns.tolist()

# =========================================================
# FORMULATION GROUPS (R2.4 / R2.8)
# A formulation group = one experimentally distinct concrete
# mixture: unique combination of the categorical mix descriptors
# and the design compressive strength.
# =========================================================

group_cols = CAT_COLS + ["Design_F'c (Mpa)"]
group_cols = [c for c in group_cols if c in df.columns]
groups = df[group_cols].astype(str).agg("|".join, axis=1)
group_ids = pd.factorize(groups)[0]
n_groups = len(np.unique(group_ids))
group_sizes = pd.Series(group_ids).value_counts()

log(f"\nFormulation groups: {n_groups}")
log(f"Group sizes: min={group_sizes.min()} median={group_sizes.median()} max={group_sizes.max()}")

pd.DataFrame({
    "group_id": group_sizes.index,
    "n_samples": group_sizes.values
}).to_csv(os.path.join(TABLES_DIR, "formulation_group_sizes.csv"), index=False)

# Redundancy analysis (R2.8)
feat_cols = [c for c in df.columns if c != TARGET]
dup_features_only = int(df.duplicated(subset=feat_cols).sum())
dup_full_rows = int(df.duplicated().sum())
combo_cols = group_cols + ["Curing_age_(days)"]
n_unique_combo = int(df[combo_cols].astype(str).agg("|".join, axis=1).nunique())

within_var = df.groupby(group_ids)[TARGET].var().mean()
global_var = df[TARGET].var()

redundancy = {
    "n_samples": int(len(df)),
    "n_formulation_groups": int(n_groups),
    "n_unique_formulation_x_age": n_unique_combo,
    "duplicate_rows_features_only": dup_features_only,
    "duplicate_rows_full": dup_full_rows,
    "mean_within_group_target_variance": float(within_var),
    "global_target_variance": float(global_var),
    "variance_ratio_within_over_global": float(within_var / global_var),
}
with open(os.path.join(TABLES_DIR, "redundancy_analysis.json"), "w") as f:
    json.dump(redundancy, f, indent=2)
log("\nRedundancy analysis:\n" + json.dumps(redundancy, indent=2))

# =========================================================
# EXPLORATORY IMPUTATION BENCHMARK (NOT USED TO SELECT THE CV IMPUTERS)
# This full-data summary is descriptive only. The actual model-CV imputer
# selection is performed independently within each outer training fold below.
# Leakage-aware: predictors used to impute Ts or Fs exclude BOTH
# the main prediction target Cs AND the sibling strength variable,
# so no target-related mechanical information enters the imputed
# values. Selection criterion (defined a priori, applied to both
# variables identically): lowest cross-validated RMSE_mean;
# ties broken by lower MAE_mean.
# =========================================================

log("\nImputation evaluation (leakage-aware predictor sets)")
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
missing_targets = ["Ts_(Mpa)", "Fs_(Mpa)"]
STRENGTH_VARS = ["Ts_(Mpa)", "Fs_(Mpa)", TARGET]

imputation_results = []
for imp_target in missing_targets:
    complete_df = df[df[imp_target].notnull()].copy()
    banned = [c for c in STRENGTH_VARS if c in complete_df.columns]
    X_imp = complete_df.drop(columns=banned)
    y_imp = complete_df[imp_target]

    imputation_models = {
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    }
    for model_name, model in imputation_models.items():
        r2_list, rmse_list, mae_list = [], [], []
        for train_idx, test_idx in kf.split(X_imp):
            X_tr, X_te = X_imp.iloc[train_idx], X_imp.iloc[test_idx]
            y_tr, y_te = y_imp.iloc[train_idx], y_imp.iloc[test_idx]
            prep = build_preprocessor(X_tr)
            X_tr_enc = prep.fit_transform(X_tr)
            X_te_enc = prep.transform(X_te)
            model.fit(X_tr_enc, y_tr)
            pred = model.predict(X_te_enc)
            r2_list.append(r2_score(y_te, pred))
            rmse_list.append(rmse(y_te, pred))
            mae_list.append(mean_absolute_error(y_te, pred))
        imputation_results.append({
            "target": imp_target, "model": model_name,
            "R2_mean": np.mean(r2_list), "R2_std": np.std(r2_list),
            "RMSE_mean": np.mean(rmse_list), "RMSE_std": np.std(rmse_list),
            "MAE_mean": np.mean(mae_list), "MAE_std": np.std(mae_list),
        })

imputation_df = pd.DataFrame(imputation_results)
imputation_df.to_csv(os.path.join(TABLES_DIR, "imputation_results.csv"), index=False)
log(imputation_df.to_string(index=False))

# Apply the a-priori criterion
selected_imputer = {}
for imp_target in missing_targets:
    sub = imputation_df[imputation_df["target"] == imp_target]
    sub = sub.sort_values(["RMSE_mean", "MAE_mean"])
    selected_imputer[imp_target] = sub.iloc[0]["model"]
log(f"\nSelected imputers by criterion (min RMSE, tie-break MAE): {selected_imputer}")
with open(os.path.join(TABLES_DIR, "exploratory_full_data_imputer_choice_NOT_USED_FOR_CV.json"), "w") as f:
    json.dump(selected_imputer, f, indent=2)


def make_imputer(name):
    if name == "KNN":
        return KNeighborsRegressor(n_neighbors=5)
    return RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)


def select_imputers_on_training(train_df, n_splits=5):
    """Select Ts/Fs imputation algorithms using ONLY one outer-training fold.

    This selection is deliberately nested inside each outer split. Its CV is
    random row-wise because the task is the conventional random-CV experiment;
    it must not be interpreted as evidence of formulation-level generalization.
    """
    selected = {}
    for imp_target in missing_targets:
        complete = train_df[train_df[imp_target].notna()].copy()
        banned = [c for c in STRENGTH_VARS if c in complete.columns]
        X_imp = complete.drop(columns=banned)
        y_imp = complete[imp_target]
        n_inner = min(n_splits, len(complete))
        if n_inner < 2:
            raise ValueError(f"Not enough observed values to select imputer for {imp_target}.")
        inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for model_name in ["KNN", "RandomForest"]:
            rmses, maes = [], []
            for itr, iva in inner_cv.split(X_imp):
                Xtr, Xva = X_imp.iloc[itr], X_imp.iloc[iva]
                ytr, yva = y_imp.iloc[itr], y_imp.iloc[iva]
                prep = build_preprocessor(Xtr)
                Xtr_e = prep.fit_transform(Xtr)
                Xva_e = prep.transform(Xva)
                model = make_imputer(model_name)
                model.fit(Xtr_e, ytr)
                pred = model.predict(Xva_e)
                rmses.append(rmse(yva, pred))
                maes.append(mean_absolute_error(yva, pred))
            scores.append((float(np.mean(rmses)), float(np.mean(maes)), model_name))
        selected[imp_target] = sorted(scores)[0][2]
    return selected


def impute_fold(train_df, test_df, fold_selected_imputer):
    """Fit imputers on outer-training rows only; apply to train/test partitions."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    for imp_target in missing_targets:
        banned = [c for c in STRENGTH_VARS if c in train_df.columns]
        fit_rows = train_df[train_df[imp_target].notnull()]
        X_fit = fit_rows.drop(columns=banned)
        y_fit = fit_rows[imp_target]
        prep = build_preprocessor(X_fit)
        X_fit_enc = prep.fit_transform(X_fit)
        model = make_imputer(fold_selected_imputer[imp_target])
        model.fit(X_fit_enc, y_fit)
        for part in (train_df, test_df):
            miss = part[imp_target].isnull()
            if miss.any():
                X_miss = part.loc[miss].drop(columns=banned)
                part.loc[miss, imp_target] = model.predict(prep.transform(X_miss))
    return train_df, test_df


# =========================================================
# MODELS (single fixed configuration for every experiment)
# =========================================================

def make_models():
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "SVR_RBF": SVR(kernel="rbf"),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, objective="reg:squarederror"),
        "LightGBM": LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbose=-1),
        "CatBoost": CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=5,
            verbose=0, random_state=RANDOM_STATE),
    }


experiments = {
    "Exp1_with_Ts_Fs": [],
    "Exp2_without_Ts_Fs": ["Ts_(Mpa)", "Fs_(Mpa)"],
    "Exp3_without_Ts_Fs_DesignFc": ["Ts_(Mpa)", "Fs_(Mpa)", "Design_F'c (Mpa)"],
}

# =========================================================
# MAIN EXPERIMENTS: nested 5-fold CV (R1.5, R1.6, R2.5)
# =========================================================

log("\nMain experiments (nested 5-fold CV)")
all_results = []
fold_level = []
main_oof_rows = []
fold_imputer_choices = []

# export the fixed split indices once (identical for every experiment/model)
split_rows = []
for fold, (tr, te) in enumerate(kf.split(df), start=1):
    for i in tr:
        split_rows.append({"fold": fold, "role": "train", "row_index": int(i)})
    for i in te:
        split_rows.append({"fold": fold, "role": "test", "row_index": int(i)})
pd.DataFrame(split_rows).to_csv(os.path.join(SPLITS_DIR, "kfold_split_indices.csv"), index=False)

# Choose imputers independently within each outer training fold, once per fold.
fold_selected_imputers = {}
for _fold, (_tr, _te) in enumerate(kf.split(df), start=1):
    fold_selected_imputers[_fold] = select_imputers_on_training(df.iloc[_tr])
    for _target, _choice in fold_selected_imputers[_fold].items():
        fold_imputer_choices.append({"fold": _fold, "target": _target, "selected_imputer": _choice})
pd.DataFrame(fold_imputer_choices).to_csv(os.path.join(TABLES_DIR, "fold_specific_imputer_choices.csv"), index=False)

for exp_name, drop_cols in experiments.items():
    log(f"\n--- {exp_name} ---")
    needs_imputation = len(drop_cols) == 0  # Ts/Fs only used in Exp1
    for model_name in make_models():
        r2s, rmses, maes = [], [], []
        for fold, (train_idx, test_idx) in enumerate(kf.split(df), start=1):
            tr_df, te_df = df.iloc[train_idx], df.iloc[test_idx]
            if needs_imputation:
                tr_df, te_df = impute_fold(tr_df, te_df, fold_selected_imputers[fold])
            X_tr = tr_df.drop(columns=[TARGET] + drop_cols)
            y_tr = tr_df[TARGET]
            X_te = te_df.drop(columns=[TARGET] + drop_cols)
            y_te = te_df[TARGET]
            prep = build_preprocessor(X_tr)
            X_tr_enc = prep.fit_transform(X_tr)
            X_te_enc = prep.transform(X_te)
            model = make_models()[model_name]
            model.fit(X_tr_enc, y_tr)
            pred = model.predict(X_te_enc)
            r2v, rv, mv = r2_score(y_te, pred), rmse(y_te, pred), mean_absolute_error(y_te, pred)
            r2s.append(r2v); rmses.append(rv); maes.append(mv)
            fold_level.append({"experiment": exp_name, "model": model_name,
                               "fold": fold, "R2": r2v, "RMSE": rv, "MAE": mv})
            for _idx, _yt, _yp in zip(te_df.index, y_te.to_numpy(), np.asarray(pred)):
                main_oof_rows.append({"protocol": "Random5Fold", "experiment": exp_name,
                                      "model": model_name, "fold": fold,
                                      "row_index": int(_idx), "group_id": int(group_ids[_idx]),
                                      "y_true": float(_yt), "y_pred": float(_yp)})
        all_results.append({
            "experiment": exp_name, "model": model_name,
            "R2_mean": np.mean(r2s), "R2_std": np.std(r2s),
            "RMSE_mean": np.mean(rmses), "RMSE_std": np.std(rmses),
            "MAE_mean": np.mean(maes), "MAE_std": np.std(maes),
        })
        log(f"{model_name:16s} R2={np.mean(r2s):.6f}  RMSE={np.mean(rmses):.6f}  MAE={np.mean(maes):.6f}")

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(TABLES_DIR, "main_experiments_cv.csv"), index=False)
pd.DataFrame(fold_level).to_csv(os.path.join(TABLES_DIR, "main_experiments_fold_level.csv"), index=False)
main_oof_df = pd.DataFrame(main_oof_rows)
main_oof_df.to_csv(os.path.join(TABLES_DIR, "main_experiments_oof_predictions.csv"), index=False)

def summarize_oof_predictions(oof_df, group_keys, out_path):
    """Compute pooled OOF metrics from one prediction per held-out row."""
    rows = []
    for keys, part in oof_df.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        y_true = part["y_true"].to_numpy()
        y_pred = part["y_pred"].to_numpy()
        row = {k: v for k, v in zip(group_keys, keys)}
        row.update({"n_oof_predictions": int(len(part)),
                    "R2_pooled": float(r2_score(y_true, y_pred)),
                    "RMSE_pooled": rmse(y_true, y_pred),
                    "MAE_pooled": float(mean_absolute_error(y_true, y_pred))})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)

summarize_oof_predictions(main_oof_df, ["experiment", "model"],
                          os.path.join(TABLES_DIR, "main_experiments_oof_pooled_metrics.csv"))

# =========================================================
# GROUP-WISE VALIDATION (R2.4): GroupKFold on formulation groups
# =========================================================

log("\nGroup-wise validation (GroupKFold, 5 folds, formulation groups)")
gkf = GroupKFold(n_splits=N_SPLITS)
group_results = []
group_fold_level = []
group_oof_rows = []

group_split_rows = []
for fold, (tr, te) in enumerate(gkf.split(df, df[TARGET], group_ids), start=1):
    for i in tr:
        group_split_rows.append({"fold": fold, "role": "train", "row_index": int(i)})
    for i in te:
        group_split_rows.append({"fold": fold, "role": "test", "row_index": int(i)})
pd.DataFrame(group_split_rows).to_csv(os.path.join(SPLITS_DIR, "groupkfold_split_indices.csv"), index=False)

for exp_name in ["Exp2_without_Ts_Fs", "Exp3_without_Ts_Fs_DesignFc"]:
    drop_cols = experiments[exp_name]
    log(f"\n--- Group-wise {exp_name} ---")
    for model_name in make_models():
        r2s, rmses, maes = [], [], []
        for fold, (train_idx, test_idx) in enumerate(gkf.split(df, df[TARGET], group_ids), start=1):
            tr_df, te_df = df.iloc[train_idx], df.iloc[test_idx]
            X_tr = tr_df.drop(columns=[TARGET] + drop_cols)
            y_tr = tr_df[TARGET]
            X_te = te_df.drop(columns=[TARGET] + drop_cols)
            y_te = te_df[TARGET]
            prep = build_preprocessor(X_tr)
            X_tr_enc = prep.fit_transform(X_tr)
            X_te_enc = prep.transform(X_te)
            model = make_models()[model_name]
            model.fit(X_tr_enc, y_tr)
            pred = model.predict(X_te_enc)
            r2v, rv, mv = r2_score(y_te, pred), rmse(y_te, pred), mean_absolute_error(y_te, pred)
            r2s.append(r2v); rmses.append(rv); maes.append(mv)
            group_fold_level.append({"experiment": exp_name, "model": model_name,
                                     "fold": fold, "R2": r2v, "RMSE": rv, "MAE": mv})
            for _idx, _yt, _yp in zip(te_df.index, y_te.to_numpy(), np.asarray(pred)):
                group_oof_rows.append({"protocol": "GroupKFold5", "experiment": exp_name,
                                       "model": model_name, "fold": fold,
                                       "row_index": int(_idx), "group_id": int(group_ids[_idx]),
                                       "y_true": float(_yt), "y_pred": float(_yp)})
        group_results.append({
            "experiment": exp_name, "model": model_name,
            "R2_mean": np.mean(r2s), "R2_std": np.std(r2s),
            "RMSE_mean": np.mean(rmses), "RMSE_std": np.std(rmses),
            "MAE_mean": np.mean(maes), "MAE_std": np.std(maes),
        })
        log(f"{model_name:16s} R2={np.mean(r2s):.4f}  RMSE={np.mean(rmses):.4f}  MAE={np.mean(maes):.4f}")

pd.DataFrame(group_results).to_csv(os.path.join(TABLES_DIR, "groupwise_cv.csv"), index=False)
pd.DataFrame(group_fold_level).to_csv(os.path.join(TABLES_DIR, "groupwise_fold_level.csv"), index=False)
pd.DataFrame(group_oof_rows).to_csv(os.path.join(TABLES_DIR, "groupwise_oof_predictions.csv"), index=False)


# Leave-one-formulation-out for the two best ensembles (if group count is tractable)
if n_groups <= 120:
    log("\nLeave-one-formulation-out (XGBoost, RandomForest; Exp2 features)")
    logo = LeaveOneGroupOut()
    drop_cols = experiments["Exp2_without_Ts_Fs"]
    logo_rows = []
    for model_name in ["XGBoost", "RandomForest"]:
        y_true_all, y_pred_all = [], []
        per_group = []
        for train_idx, test_idx in logo.split(df, df[TARGET], group_ids):
            tr_df, te_df = df.iloc[train_idx], df.iloc[test_idx]
            X_tr = tr_df.drop(columns=[TARGET] + drop_cols)
            y_tr = tr_df[TARGET]
            X_te = te_df.drop(columns=[TARGET] + drop_cols)
            y_te = te_df[TARGET]
            prep = build_preprocessor(X_tr)
            X_tr_enc = prep.fit_transform(X_tr)
            X_te_enc = prep.transform(X_te)
            model = make_models()[model_name]
            model.fit(X_tr_enc, y_tr)
            pred = model.predict(X_te_enc)
            y_true_all.extend(y_te.values)
            y_pred_all.extend(pred)
            gid = group_ids[test_idx[0]]
            per_group.append({"model": model_name, "group_id": int(gid),
                              "n": len(y_te),
                              "RMSE": rmse(y_te, pred),
                              "MAE": mean_absolute_error(y_te, pred)})
        y_true_all = np.asarray(y_true_all)
        y_pred_all = np.asarray(y_pred_all)
        overall = {
            "model": model_name,
            "R2_pooled": r2_score(y_true_all, y_pred_all),
            "RMSE_pooled": rmse(y_true_all, y_pred_all),
            "MAE_pooled": mean_absolute_error(y_true_all, y_pred_all),
        }
        logo_rows.append(overall)
        log(f"LOGO {model_name}: R2={overall['R2_pooled']:.4f} RMSE={overall['RMSE_pooled']:.4f} MAE={overall['MAE_pooled']:.4f}")
        pd.DataFrame(per_group).to_csv(
            os.path.join(TABLES_DIR, f"logo_per_group_{model_name}.csv"), index=False)
    pd.DataFrame(logo_rows).to_csv(os.path.join(TABLES_DIR, "logo_overall.csv"), index=False)
else:
    log(f"\nSkipping LOGO: {n_groups} groups is too many for per-group refits")

# =========================================================
# SENSOR-ONLY MEASUREMENT PERTURBATION ROBUSTNESS
# Noise is applied only to Er and UPV, not curing age/design strength.
# Repeated perturbations use independent, reproducible seeds.
# =========================================================

log("\nSensor-only perturbation robustness (Exp2 features)")
noise_levels = [0.0, 0.01, 0.05]
N_PERTURB_REPEATS = 5
SENSOR_COLS = [c for c in ["Er_(ohm-cm)", "UPV_(m/s)"] if c in df.columns]
drop_cols = experiments["Exp2_without_Ts_Fs"]
perturb_rows = []
for model_name in ["XGBoost", "RandomForest"]:
    for noise in noise_levels:
        all_y, all_pred = [], []
        fold_metrics = []
        for repeat in range(N_PERTURB_REPEATS):
            rng = np.random.default_rng(RANDOM_STATE + repeat)
            for fold, (train_idx, test_idx) in enumerate(kf.split(df), start=1):
                tr_df, te_df = df.iloc[train_idx], df.iloc[test_idx].copy()
                X_tr = tr_df.drop(columns=[TARGET] + drop_cols)
                y_tr = tr_df[TARGET]
                X_te = te_df.drop(columns=[TARGET] + drop_cols)
                y_te = te_df[TARGET]
                if noise > 0:
                    for c in SENSOR_COLS:
                        sd = X_tr[c].std()
                        if pd.notna(sd) and sd > 0:
                            X_te[c] = X_te[c] + rng.normal(0.0, noise * sd, size=len(X_te))
                prep = build_preprocessor(X_tr)
                X_tr_enc = prep.fit_transform(X_tr)
                X_te_enc = prep.transform(X_te)
                model = make_models()[model_name]
                model.fit(X_tr_enc, y_tr)
                pred = model.predict(X_te_enc)
                all_y.extend(y_te.to_numpy())
                all_pred.extend(np.asarray(pred))
                fold_metrics.append({"repeat": repeat + 1, "fold": fold,
                                     "R2": r2_score(y_te, pred), "RMSE": rmse(y_te, pred),
                                     "MAE": mean_absolute_error(y_te, pred)})
        perturb_rows.append({
            "model": model_name, "noise_sd_fraction": noise,
            "perturbed_features": ";".join(SENSOR_COLS),
            "n_repeats": N_PERTURB_REPEATS,
            "R2_pooled": r2_score(all_y, all_pred),
            "RMSE_pooled": rmse(all_y, all_pred),
            "MAE_pooled": mean_absolute_error(all_y, all_pred),
            "R2_fold_repeat_mean": float(np.mean([x["R2"] for x in fold_metrics])),
            "RMSE_fold_repeat_mean": float(np.mean([x["RMSE"] for x in fold_metrics])),
            "MAE_fold_repeat_mean": float(np.mean([x["MAE"] for x in fold_metrics])),
        })
        pd.DataFrame(fold_metrics).to_csv(
            os.path.join(TABLES_DIR, f"perturbation_{model_name}_noise_{str(noise).replace('.', 'p')}_folds.csv"), index=False)
        log(f"{model_name:16s} noise={noise:.2f} pooled_RMSE={perturb_rows[-1]['RMSE_pooled']:.4f}")
pd.DataFrame(perturb_rows).to_csv(os.path.join(TABLES_DIR, "perturbation_robustness_sensor_only.csv"), index=False)

# =========================================================
# DESCRIPTIVE PAIRED FOLD DIFFERENCES (not inferential significance tests)
# Five CV folds have overlapping training sets and are not independent
# experimental replications. Do not interpret these summaries as p-values.
# =========================================================

log("\nDescriptive paired fold differences (Exp2, R2 and RMSE)")
fl = pd.DataFrame(fold_level)
main = fl[fl["experiment"] == "Exp2_without_Ts_Fs"]
base = main[main["model"] == "XGBoost"].sort_values("fold")
diff_rows = []
for other in ["LightGBM", "CatBoost", "RandomForest", "SVR_RBF", "LinearRegression"]:
    comp = main[main["model"] == other].sort_values("fold")
    if len(base) == len(comp) and len(base) > 0:
        for metric in ["R2", "RMSE", "MAE"]:
            d = base[metric].to_numpy() - comp[metric].to_numpy()
            diff_rows.append({"comparison": f"XGBoost - {other}", "metric": metric,
                              "n_folds": len(d), "mean_paired_difference": float(np.mean(d)),
                              "sd_paired_difference": float(np.std(d, ddof=1)) if len(d) > 1 else np.nan,
                              "interpretation": "descriptive_only; folds are not independent"})
pd.DataFrame(diff_rows).to_csv(os.path.join(TABLES_DIR, "paired_fold_differences_descriptive.csv"), index=False)
log("\nDONE. All outputs under " + BASE_DIR)

# =========================================================
# R2 ADDITION - CONTROLLED HEAD-TO-HEAD BENCHMARK
# Reimplementation of Saeheaw (2025) PolyBayes-ABR under the
# SAME outer random/group-wise protocols and feature restrictions.
# Bayesian optimization is nested within each OUTER training fold.
# =========================================================

from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import KFold as InnerKFold
import optuna

HEAD2HEAD_DIR = os.path.join(BASE_DIR, "head_to_head")
os.makedirs(HEAD2HEAD_DIR, exist_ok=True)

# Numeric polynomial expansion + categorical OHE. Fitted on training data only.
def build_poly_preprocessor(X):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("num_poly", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", enc, cat_cols),
    ])


def make_abr(params):
    tree = DecisionTreeRegressor(
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=RANDOM_STATE,
    )
    try:
        return AdaBoostRegressor(
            estimator=tree,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=RANDOM_STATE,
        )
    except TypeError:
        return AdaBoostRegressor(
            base_estimator=tree,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=RANDOM_STATE,
        )


def tune_polybayes(X_train, y_train, groups_train=None, n_trials=50, protocol="Random5Fold", outer_fold=1, outer_train_indices=None):
    # Keep formulation groups separated in inner CV whenever the outer protocol
    # is group-wise; random KFold is used only for the random-CV protocol.
    if groups_train is not None:
        if len(np.unique(groups_train)) < 2:
            raise ValueError("Need at least two formulation groups for inner GroupKFold.")
        inner = GroupKFold(n_splits=min(5, len(np.unique(groups_train))))
    else:
        inner = InnerKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # Save exact original-dataset indices for every inner split before optimization.
    split_list = list(inner.split(X_train, y_train, groups_train) if groups_train is not None else inner.split(X_train, y_train))
    original_indices = np.asarray(X_train.index if outer_train_indices is None else outer_train_indices)
    inner_records = []
    for inner_fold, (itr, iva) in enumerate(split_list, 1):
        for role, positions in (("train", itr), ("validation", iva)):
            inner_records.extend({"protocol": protocol, "outer_fold": outer_fold,
                                  "inner_fold": inner_fold, "role": role,
                                  "row_index": int(idx)} for idx in original_indices[positions])
    pd.DataFrame(inner_records).to_csv(os.path.join(HEAD2HEAD_DIR, f"polybayes_inner_splits_{protocol}_outer{outer_fold}.csv"), index=False)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.1, 1.5),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
        losses = []
        for itr, iva in split_list:
            Xi_tr, Xi_va = X_train.iloc[itr], X_train.iloc[iva]
            yi_tr, yi_va = y_train.iloc[itr], y_train.iloc[iva]
            prep = build_poly_preprocessor(Xi_tr)
            Xi_tr_e = prep.fit_transform(Xi_tr)
            Xi_va_e = prep.transform(Xi_va)
            model = make_abr(params)
            model.fit(Xi_tr_e, yi_tr)
            pred = model.predict(Xi_va_e)
            losses.append(mean_squared_error(yi_va, pred))
        return float(np.mean(losses))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trials = study.trials_dataframe(attrs=("number", "value", "datetime_start", "datetime_complete", "duration", "params", "state"))
    trials.insert(0, "outer_fold", outer_fold)
    trials.insert(0, "protocol", protocol)
    trials.to_csv(os.path.join(HEAD2HEAD_DIR, f"polybayes_optuna_trials_{protocol}_outer{outer_fold}.csv"), index=False)
    return study.best_params, study.best_value


def evaluate_polybayes_outer(splitter, Xdf, y, groups_arg=None, protocol="Random5Fold", n_trials=POLYBAYES_TRIALS):
    rows, params_rows, oof_rows = [], [], []
    split_iter = splitter.split(Xdf, y, groups_arg) if groups_arg is not None else splitter.split(Xdf, y)
    for fold, (tr, te) in enumerate(split_iter, 1):
        Xtr, Xte = Xdf.iloc[tr], Xdf.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        groups_tr = None if groups_arg is None else np.asarray(groups_arg)[tr]
        best_params, inner_mse = tune_polybayes(Xtr, ytr, groups_train=groups_tr, n_trials=n_trials, protocol=protocol, outer_fold=fold, outer_train_indices=Xtr.index)
        prep = build_poly_preprocessor(Xtr)
        Xtr_e = prep.fit_transform(Xtr)
        Xte_e = prep.transform(Xte)
        model = make_abr(best_params)
        model.fit(Xtr_e, ytr)
        pred = model.predict(Xte_e)
        for _idx, _yt, _yp in zip(Xte.index, yte.to_numpy(), np.asarray(pred)):
            oof_rows.append({"protocol": protocol, "fold": fold, "model": "PolyBayes-ABR",
                             "row_index": int(_idx), "group_id": int(group_ids[_idx]),
                             "y_true": float(_yt), "y_pred": float(_yp)})
        rows.append({
            "protocol": protocol, "fold": fold, "model": "PolyBayes-ABR",
            "R2": r2_score(yte, pred), "RMSE": rmse(yte, pred),
            "MAE": mean_absolute_error(yte, pred),
        })
        params_rows.append({"protocol": protocol, "fold": fold,
                            "inner_best_MSE": inner_mse, **best_params})
        log(f"PolyBayes {protocol} fold={fold}: R2={rows[-1]['R2']:.4f} RMSE={rows[-1]['RMSE']:.4f}")
    return pd.DataFrame(rows), pd.DataFrame(params_rows), pd.DataFrame(oof_rows)


# Reviewer-requested admissible feature set: Exp2 excludes Ts/Fs.
H2H_DROP = experiments["Exp2_without_Ts_Fs"]
X_h2h = df.drop(columns=[TARGET] + H2H_DROP)
y_h2h = df[TARGET]

log("\nHEAD-TO-HEAD: PolyBayes-ABR, Exp2 admissible features")
poly_random, params_random, oof_poly_random = evaluate_polybayes_outer(
    KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
    X_h2h, y_h2h, protocol="Random5Fold", n_trials=POLYBAYES_TRIALS)
poly_group, params_group, oof_poly_group = evaluate_polybayes_outer(
    GroupKFold(n_splits=N_SPLITS), X_h2h, y_h2h,
    groups_arg=group_ids, protocol="GroupKFold5", n_trials=POLYBAYES_TRIALS)

poly_folds = pd.concat([poly_random, poly_group], ignore_index=True)
poly_params = pd.concat([params_random, params_group], ignore_index=True)
poly_folds.to_csv(os.path.join(HEAD2HEAD_DIR, "polybayes_fold_level.csv"), index=False)
poly_params.to_csv(os.path.join(HEAD2HEAD_DIR, "polybayes_best_params_by_outer_fold.csv"), index=False)
poly_oof_all = pd.concat([oof_poly_random, oof_poly_group], ignore_index=True)
poly_oof_all.to_csv(os.path.join(HEAD2HEAD_DIR, "polybayes_oof_predictions.csv"), index=False)
summarize_oof_predictions(poly_oof_all, ["protocol", "model"],
                          os.path.join(HEAD2HEAD_DIR, "polybayes_oof_pooled_metrics.csv"))

poly_summary = poly_folds.groupby(["protocol", "model"], as_index=False).agg(
    R2_mean=("R2", "mean"), R2_std=("R2", "std"),
    RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
    MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"))
poly_summary.to_csv(os.path.join(HEAD2HEAD_DIR, "polybayes_summary.csv"), index=False)
log("\nPolyBayes controlled benchmark summary:\n" + poly_summary.to_string(index=False))

# Controlled ablation table requested by Reviewer 3: conventional random CV vs group-wise CV.
# Add the already-computed XGBoost reference under exactly the same Exp2 feature restriction.
existing_random = pd.DataFrame(fold_level)
existing_random = existing_random[(existing_random.experiment == "Exp2_without_Ts_Fs") & (existing_random.model == "XGBoost")].copy()
existing_random["protocol"] = "Random5Fold"
existing_group = pd.DataFrame(group_fold_level)
existing_group = existing_group[(existing_group.experiment == "Exp2_without_Ts_Fs") & (existing_group.model == "XGBoost")].copy()
existing_group["protocol"] = "GroupKFold5"
xgb_h2h = pd.concat([existing_random, existing_group], ignore_index=True)[["protocol","fold","model","R2","RMSE","MAE"]]
controlled = pd.concat([xgb_h2h, poly_folds], ignore_index=True)
controlled.to_csv(os.path.join(HEAD2HEAD_DIR, "controlled_head_to_head_fold_level.csv"), index=False)
controlled_summary = controlled.groupby(["protocol","model"], as_index=False).agg(
    R2_mean=("R2","mean"), R2_std=("R2","std"),
    RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"),
    MAE_mean=("MAE","mean"), MAE_std=("MAE","std"))
controlled_summary.to_csv(os.path.join(HEAD2HEAD_DIR, "controlled_head_to_head_summary.csv"), index=False)
log("\nCONTROLLED HEAD-TO-HEAD SUMMARY:\n" + controlled_summary.to_string(index=False))


# Run manifest: records the protocol and key caveats for reproducibility.
run_manifest = {
    "script": os.path.basename(__file__),
    "data_file": os.path.basename(DATA_PATH),
    "target": TARGET,
    "random_state": RANDOM_STATE,
    "n_splits": N_SPLITS,
    "polybayes_trials_per_outer_fold": POLYBAYES_TRIALS,
    "n_formulation_groups": int(n_groups),
    "group_definition_columns": group_cols,
    "categorical_whitespace_trimmed_in_memory": True,
    "imputer_selection": "within each outer random-CV training fold; inner KFold",
    "polybayes_inner_cv": "GroupKFold when outer protocol is GroupKFold; KFold for random protocol",
    "perturbation_features": SENSOR_COLS,
    "perturbation_repeats": N_PERTURB_REPEATS,
    "statistical_tests": "no inferential test on overlapping CV folds; descriptive paired differences only",
    "note": "This script does not calculate SHAP. SHAP must be regenerated from explicitly saved fitted models/background/evaluation rows in a separate, documented analysis."
}
with open(os.path.join(BASE_DIR, "run_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(run_manifest, f, indent=2, ensure_ascii=False)
