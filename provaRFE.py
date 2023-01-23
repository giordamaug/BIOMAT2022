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

x = pd.read_csv("KIDNEY/node_attributes.csv", index_col=0)
y = pd.read_csv("KIDNEY/node_labels.csv", index_col=0)
genes = y.index.to_list()

#y["label"] = y.apply(lambda row: 0 if row["most_freq"] in ["aE", "aNE", "NE"] else 1,axis=1)
y["label"] = y.apply(lambda row: 0 if row["most_freq"] in ["E"] else 1 if row["most_freq"] in ["NE"] else -1,axis=1)
y = y[y['label'] >= 0]
x = pd.concat([x, y["label"]], axis=1, join="inner")
for col in x.columns[x.isna().any()].tolist():
    mean_value=x[col].mean()          # Replace NaNs in column with the mean of values in the same column
    if mean_value is not np.nan:
        x[col].fillna(value=mean_value, inplace=True)
    else:                             # otherwise, if the mean is NaN, remove the column
        x = x.drop(col, 1)

print(y)
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
import lightgbm as lgb
from tabulate import tabulate

y = x["label"].values
x = x.drop(columns=["label"]).to_numpy()
if True:
    estimator = RandomForestClassifier()
    selector = RFE(estimator, n_features_to_select=10, step=1, verbose=1)
    x = selector.fit_transform(x, y)

seed = 0
set_seed(0)
nfolds = 5
kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
accuracies, mccs = [], []

classes = np.unique(y)
nclasses = len(classes)
cma = np.zeros(shape=(nclasses,nclasses), dtype=np.int)
mm = np.array([], dtype=np.int)
yy = np.array([], dtype=np.int)
predictions = np.array([])
columns_names = ["Accuracy","BA", "Sensitivity", "Specificity","MCC", 'CM']
scores = pd.DataFrame(columns=columns_names)
print(f'Classification with method LGBM...')
for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(np.arange(len(x)), y), total=kf.get_n_splits(), desc= f"{nfolds}-fold")):
    train_x, train_y, test_x, test_y = x[train_idx], y[train_idx], x[test_idx], y[test_idx],
    mm = np.concatenate((mm, test_idx))
    yy = np.concatenate((yy, test_y))
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
print(tabulate(df_scores, headers='keys', tablefmt='psql'))