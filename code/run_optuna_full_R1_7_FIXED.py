"""Full 5x50 Optuna audit for Random5Fold and GroupKFold5, original 4420 rows."""
import ast,os,sys,json,time
from pathlib import Path
import numpy as np,pandas as pd,optuna
from sklearn.model_selection import KFold,GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'concrete_xai_pipeline_R3_with_optuna_logging.py'
DATA = ROOT / 'Data(4).csv'
if not DATA.exists():
    DATA = ROOT / 'Data.csv'
if not DATA.exists():
    raise FileNotFoundError('ضع Data(4).csv أو Data.csv بنفس مجلد السكربت')
HEAD2HEAD_DIR = str(ROOT / 'R1_7_FULL_RUN' / 'head_to_head')
os.makedirs(HEAD2HEAD_DIR,exist_ok=True)
RANDOM_STATE=42
POLYBAYES_TRIALS=50
InnerKFold=KFold
TARGET='Cs_(Mpa)'
def rmse(y,p):return float(np.sqrt(mean_squared_error(y,p)))
def log(msg): print(msg, flush=True)
tree=ast.parse(SOURCE.read_text())
names={'build_poly_preprocessor','make_abr','tune_polybayes','evaluate_polybayes_outer'}
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
exec(compile(ast.Module(body=selected,type_ignores=[]),str(SOURCE),'exec'))
df=pd.read_csv(DATA).drop(columns=['Unnamed: 0'],errors='ignore')
for c in df.select_dtypes('object').columns:df[c]=df[c].str.strip()
assert df.shape==(4420,11),df.shape
X=df.drop(columns=[TARGET,'Ts_(Mpa)','Fs_(Mpa)']);y=df[TARGET]
group_cols=['Type_of_cement','Brand','Additives','Type_of_aggregates',"Design_F'c (Mpa)"]
group_ids=pd.factorize(pd.MultiIndex.from_frame(df[group_cols]))[0]
assert len(np.unique(group_ids))==12
start=time.time()
for protocol,splitter,groups in [('Random5Fold',KFold(5,shuffle=True,random_state=42),None),('GroupKFold5',GroupKFold(5),group_ids)]:
    print('START',protocol,flush=True)
    f,p,o=evaluate_polybayes_outer(splitter,X,y,groups_arg=groups,protocol=protocol,n_trials=50)
    f.to_csv(f'{HEAD2HEAD_DIR}/polybayes_fold_level_{protocol}.csv',index=False)
    p.to_csv(f'{HEAD2HEAD_DIR}/polybayes_best_params_{protocol}.csv',index=False)
    o.to_csv(f'{HEAD2HEAD_DIR}/polybayes_oof_{protocol}.csv',index=False)
    print('FINISHED',protocol,'seconds',round(time.time()-start),flush=True)
print('ALL COMPLETE',round(time.time()-start),flush=True)
