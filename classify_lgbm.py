import warnings
warnings.filterwarnings('ignore')
import random
import pandas as pd
import numpy as np
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)

import sys
from tqdm import tqdm
import pandas as pd
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import *
import lightgbm as lgb
from tabulate import tabulate

seed=0
df = pd.read_csv("KIDNEY/kidney_EvsNE.csv", index_col=0)
set_seed(seed)
nfolds = 5
kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
accuracies, mccs = [], []
genes = df.index.values
X = df.loc[:, df.columns != 'class'].to_numpy()
y = df['class'].values

params1 = {'boosting_type': 'gbdt', 'class_weight': 'balanced', 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.1, 'max_depth': -1, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 100, 'n_jobs': -1, 'num_leaves': 31, 'objective': 'multiclass', 'num_class': 2, 'random_state': 0, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'silent': 'warn', 'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0}
params2 = { 'boosting': 'gbdt', 'objective': 'binary', 'num_leaves': 31, 'random_state': seed, 'verbose' : -1, 'scale_pos_weight':0.1} 

def focal_loss_lgb_f1_score(preds, lgbDataset):
  preds = 1/(1+np.exp(-preds))  # sigmoid
  binary_preds = [int(p>0.5) for p in preds]
  y_true = lgbDataset.get_label()
  return 'f1', f1_score(y_true, binary_preds), True

classes = np.unique(y)
nclasses = len(classes)
cma = np.zeros(shape=(nclasses,nclasses), dtype=np.int)
mm = np.array([], dtype=np.int)
gg = np.array([])
yy = np.array([], dtype=np.int)
predictions = np.array([])
columns_names = ["Accuracy","BA", "Sensitivity", "Specificity","MCC", 'CM']
scores = pd.DataFrame(columns=columns_names)
print(bcolors.HEADER + f'Classification with method LGBM...' + bcolors.ENDC)
for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(np.arange(len(X)), y), total=kf.get_n_splits(), desc=bcolors.OKGREEN +  f"{nfolds}-fold")):
    train_x, train_y, test_x, test_y = X[train_idx], y[train_idx], X[test_idx], y[test_idx],
    mm = np.concatenate((mm, test_idx))
    yy = np.concatenate((yy, test_y))
    gg = np.concatenate((gg, genes[test_idx]))
    if True:
    	clf = lgb.LGBMClassifier(random_state=seed, class_weight="balanced")
    	clf.fit(train_x, train_y)
    	preds =clf.predict(test_x)
    else:
    	lgb_train = lgb.Dataset(train_x, train_y)
    	lgb_eval = lgb.Dataset(test_x, test_y, reference=lgb_train)
    	clf = lgb.train(params2, train_set=lgb_train)# , valid_sets=lgb_eval, early_stopping_rounds=30)
    	preds = clf.predict(test_x)
    	preds = np.where(preds > 0.5, 1, 0)    # for binary
    	#preds = np.argmax(preds, axis=1)
    cm = confusion_matrix(test_y, preds)
    cma += cm.astype(int)
    predictions = np.concatenate((predictions, preds))
    scores = scores.append(pd.DataFrame([[accuracy_score(test_y, preds), balanced_accuracy_score(test_y, preds), 
        cm[0,0]/(cm[0,0]+cm[0,1]), cm[1,1]/(cm[1,0]+cm[1,1]), 
        matthews_corrcoef(test_y, preds), cm]], columns=columns_names, index=[fold]))
dfm_scores = pd.DataFrame(scores.mean(axis=0)).T
dfs_scores = pd.DataFrame(scores.std(axis=0)).T
df_scores = pd.DataFrame([f'{row[0]:.3f}±{row[1]:.3f}' for row in pd.concat([dfm_scores,dfs_scores], axis=0).T.values.tolist()]).T
df_scores.index=['LGBM']
df_scores['CM'] = [cma]
df_scores.columns = columns_names
print(bcolors.OKGREEN +  tabulate(df_scores, headers='keys', tablefmt='psql') + bcolors.ENDC)
 