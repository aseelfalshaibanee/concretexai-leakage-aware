# =========================================================
# ConcreteXAI Pipeline - R1 REVISED (IJIES Paper ID 20265490)
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
warnings.filterwarnings("ignore")

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

from scipy.stats import wilcoxon

# =========================================================
# SETTINGS
# =========================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Data.csv")
BASE_DIR = os.path.join(PROJECT_ROOT, "results")
TARGET = "Cs_(Mpa)"
RANDOM_STATE = 42
N_SPLITS = 5

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
# IMPUTATION EVALUATION (R1.8 / R2.5)
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
with open(os.path.join(TABLES_DIR, "selected_imputers.json"), "w") as f:
    json.dump(selected_imputer, f, indent=2)


def make_imputer(name):
    if name == "KNN":
        return KNeighborsRegressor(n_neighbors=5)
    return RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)


def impute_fold(train_df, test_df):
    """Fit imputers for Ts/Fs on the training partition only and apply to
    both partitions (fully nested, R2.5). Predictors exclude all strength
    variables (leakage-aware)."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    for imp_target in missing_targets:
        banned = [c for c in STRENGTH_VARS if c in train_df.columns]
        fit_rows = train_df[train_df[imp_target].notnull()]
        X_fit = fit_rows.drop(columns=banned)
        y_fit = fit_rows[imp_target]
        prep = build_preprocessor(X_fit)
        X_fit_enc = prep.fit_transform(X_fit)
        model = make_imputer(selected_imputer[imp_target])
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

# export the fixed split indices once (identical for every experiment/model)
split_rows = []
for fold, (tr, te) in enumerate(kf.split(df), start=1):
    for i in tr:
        split_rows.append({"fold": fold, "role": "train", "row_index": int(i)})
    for i in te:
        split_rows.append({"fold": fold, "role": "test", "row_index": int(i)})
pd.DataFrame(split_rows).to_csv(os.path.join(SPLITS_DIR, "kfold_split_indices.csv"), index=False)

for exp_name, drop_cols in experiments.items():
    log(f"\n--- {exp_name} ---")
    needs_imputation = len(drop_cols) == 0  # Ts/Fs only used in Exp1
    for model_name in make_models():
        r2s, rmses, maes = [], [], []
        for fold, (train_idx, test_idx) in enumerate(kf.split(df), start=1):
            tr_df, te_df = df.iloc[train_idx], df.iloc[test_idx]
            if needs_imputation:
                tr_df, te_df = impute_fold(tr_df, te_df)
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

# =========================================================
# GROUP-WISE VALIDATION (R2.4): GroupKFold on formulation groups
# =========================================================

log("\nGroup-wise validation (GroupKFold, 5 folds, formulation groups)")
gkf = GroupKFold(n_splits=N_SPLITS)
group_results = []
group_fold_level = []

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
        group_results.append({
            "experiment": exp_name, "model": model_name,
            "R2_mean": np.mean(r2s), "R2_std": np.std(r2s),
            "RMSE_mean": np.mean(rmses), "RMSE_std": np.std(rmses),
            "MAE_mean": np.mean(maes), "MAE_std": np.std(maes),
        })
        log(f"{model_name:16s} R2={np.mean(r2s):.4f}  RMSE={np.mean(rmses):.4f}  MAE={np.mean(maes):.4f}")

pd.DataFrame(group_results).to_csv(os.path.join(TABLES_DIR, "groupwise_cv.csv"), index=False)
pd.DataFrame(group_fold_level).to_csv(os.path.join(TABLES_DIR, "groupwise_fold_level.csv"), index=False)

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
# MEASUREMENT PERTURBATION ROBUSTNESS (R2.7)
# Gaussian noise on the numeric measurement inputs at test time.
# =========================================================

log("\nMeasurement-perturbation robustness (Exp2 features)")
rng = np.random.default_rng(RANDOM_STATE)
noise_levels = [0.0, 0.01, 0.05]
drop_cols = experiments["Exp2_without_Ts_Fs"]
perturb_rows = []
for model_name in ["XGBoost", "RandomForest"]:
    for noise in noise_levels:
        r2s, rmses, maes = [], [], []
        for train_idx, test_idx in kf.split(df):
            tr_df, te_df = df.iloc[train_idx], df.iloc[test_idx].copy()
            X_tr = tr_df.drop(columns=[TARGET] + drop_cols)
            y_tr = tr_df[TARGET]
            X_te = te_df.drop(columns=[TARGET] + drop_cols)
            y_te = te_df[TARGET]
            if noise > 0:
                num_cols = X_te.select_dtypes(exclude="object").columns
                for c in num_cols:
                    sd = X_tr[c].std()
                    X_te[c] = X_te[c] + rng.normal(0.0, noise * sd, size=len(X_te))
            prep = build_preprocessor(X_tr)
            X_tr_enc = prep.fit_transform(X_tr)
            X_te_enc = prep.transform(X_te)
            model = make_models()[model_name]
            model.fit(X_tr_enc, y_tr)
            pred = model.predict(X_te_enc)
            r2s.append(r2_score(y_te, pred))
            rmses.append(rmse(y_te, pred))
            maes.append(mean_absolute_error(y_te, pred))
        perturb_rows.append({
            "model": model_name, "noise_sd_fraction": noise,
            "R2_mean": np.mean(r2s), "RMSE_mean": np.mean(rmses), "MAE_mean": np.mean(maes),
        })
        log(f"{model_name:16s} noise={noise:.2f}  R2={np.mean(r2s):.4f}  RMSE={np.mean(rmses):.4f}")
pd.DataFrame(perturb_rows).to_csv(os.path.join(TABLES_DIR, "perturbation_robustness.csv"), index=False)

# =========================================================
# WILCOXON TESTS on fold-level R2 (Exp2)
# =========================================================

log("\nWilcoxon signed-rank tests (Exp2, fold-level R2)")
fl = pd.DataFrame(fold_level)
main = fl[fl["experiment"] == "Exp2_without_Ts_Fs"]
base = main[main["model"] == "XGBoost"].sort_values("fold")["R2"].values
wil_rows = []
for other in ["LightGBM", "CatBoost", "RandomForest"]:
    comp = main[main["model"] == other].sort_values("fold")["R2"].values
    try:
        stat, p = wilcoxon(base, comp)
    except Exception:
        stat, p = np.nan, np.nan
    wil_rows.append({"comparison": f"XGBoost vs {other}", "wilcoxon_stat": stat, "p_value": p})
    log(f"XGBoost vs {other}: p={p}")
pd.DataFrame(wil_rows).to_csv(os.path.join(TABLES_DIR, "wilcoxon_tests.csv"), index=False)

log("\nDONE. All outputs under " + BASE_DIR)
