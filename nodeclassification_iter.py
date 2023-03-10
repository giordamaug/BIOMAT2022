# -*- coding: utf-8 -*-
""" node classification

python nodeclassification_4paper_repeat_par.py -d NEW/LUNG 
-A "{'CS0': 'E', 'CS1':'aE', 'CS2': 'aNE', 'CS3': 'aNE','CS4': 'aNE','CS5': 'aNE','CS6': 'NE','CS7': 'NE','CS8': 'NE','CS9': 'NE'}" 
-a nodes_lung_bioattr_subloc_3305CC.csv nodes_lung_BPBeder.csv nodes_lung_CCBeder.csv 
-F nodes_lung_bioattr_subloc_3305CC.csv -l label_wo000916 
-m LGBM -O -x ND aE aNE 
-I mean -Z zscore

"""

import sys,os
from ast import literal_eval

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
    
import argparse
from operator import index
import numpy as np
import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
        
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.neural_network import MLPClassifier
import sys
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import *
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier
from logitboost import LogitBoost
from tabulate import tabulate
from progress.counter import Stack
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='BIOMAT 2022 Workbench')
parser.add_argument('-i', "--inputfile", dest='inputfile', metavar='<inputfile>', nargs="+", type=str, help='input attribute filename list', required=True)
parser.add_argument('-X', "--excludelabels", dest='excludelabels', metavar='<excludelabels>', nargs="+", default=[], help='labels to exclude (default NaN, values any list)', required=False)
parser.add_argument('-L', "--labelname", dest='labelname', metavar='<labelname>',  type=str, help='label name (default label)', default='label', required=False)
parser.add_argument('-l', "--labelfile", dest='labelfile', metavar='<labelfile>', type=str, help='label filename', required=True)
parser.add_argument('-A', "--aliases", dest='aliases', default="{}", metavar='<aliases>', required=False)
parser.add_argument('-Z', "--normalize", dest='normalize', metavar='<normalize>', type=str, help='normalize mode (default: None, choice: None|zscore|minmax)', choices=[None, 'zscore', 'minmax'], default=None, required=False)
parser.add_argument('-I', "--imputation", dest='imputation', metavar='<imputation>', type=str, help='imputation mode (default: None, choice: None|mean|zero)', choices=[None, 'mean', 'zero'], default=None, required=False)
parser.add_argument('-b', "--seed", dest='seed', metavar='<seed>', type=int, help='seed (default: 1)' , default='1', required=False)
parser.add_argument('-r', "--repeat", dest='repeat', metavar='<repeat>', type=int, help='n. of iteration (default: 1)' , default=1, required=False)
parser.add_argument('-f', "--folds", dest='folds', metavar='<folds>', type=int, help='n. of cv folds (default: 5)' , default=5, required=False)
parser.add_argument('-j', "--jobs", dest='jobs', metavar='<jobs>', type=int, help='n. of parallel jobs (default: -1)' , default=-1, required=False)
parser.add_argument('-m', "--method", dest='method', metavar='<method>', type=str, help='classifier name (default: RF, choice: RF|SVM|XGB|LGBM|MLP|LG|EXT|VC)', choices=['VC', 'RF', 'RUS', 'SVM', 'XGB', 'LGBM', 'EXT', 'MLP', 'WNN', 'LG'], default='RF', required=False)
parser.add_argument('-Q', "--removefeat", action='store_true', required=False)
args = parser.parse_args()

seed=args.seed

# get label aliases
label_aliases = literal_eval(args.aliases)
import warnings
warnings.filterwarnings('ignore')
import random
import pandas as pd
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)

"""# Read the labels
Load the label file, select the label type, and print label distribution
"""

from sklearn import preprocessing
from collections import Counter
import matplotlib.pyplot as plt

#@title Testo del titolo predefinito { run: "auto", form-width: "20%" }
label_filename = args.labelfile #@param {type:"string"}
labelname = args.labelname #@param {type:"string"}
exclude_labels = [np.nan] + args.excludelabels #@param {type:"raw"}
label_file = os.path.join(label_filename)
df_label = pd.read_csv(label_file, sep=',', index_col=0)
if labelname in df_label.columns:
    print(bcolors.HEADER + f'Loading label {labelname} from file "{label_file}"...' + bcolors.ENDC)
else:
    print(bcolors.FAIL + f'FAIL: Label name {labelname} is not in the label file{label_filename}!' + bcolors.ENDC)
print(bcolors.OKCYAN + f'Labeling...' + bcolors.ENDC)
print(bcolors.OKGREEN + f'- working on label "{labelname}""...' + bcolors.ENDC)
dup = df_label.index.duplicated().sum()
if dup > 0:
    df_label = df_label[~df_label.index.duplicated(keep='first')]
    print(bcolors.OKGREEN + f'- removing {dup} duplicated genes...' + bcolors.ENDC)
genes = df_label.index.values                                                                    # get genes with defined labels
df_label = df_label[df_label[labelname].isin([np.nan]) == False]                # drop any row contaning NaN 
labels = np.unique(df_label[labelname].values)
for key,newkey in label_aliases.items():
    if key in labels:
        print(bcolors.OKGREEN + f'- replacing label {key} with {newkey}' + bcolors.ENDC)
        df_label = df_label.replace(key, newkey)
df_label = df_label[df_label[labelname].isin(exclude_labels) == False]                # drop any row contaning NaN 
labels = np.unique(df_label[labelname].values)
print(bcolors.OKGREEN + f'- used label values {labels} (excluded {exclude_labels})' + bcolors.ENDC)
selectedgenes = df_label.index.values
print(bcolors.OKGREEN + f'- {len(selectedgenes)} labeled genes over a total of {len(genes)}' + bcolors.ENDC)
print(bcolors.OKGREEN + f'- label distribution: {dict(df_label[labelname].value_counts())}' + bcolors.ENDC)


"""# Load attributes to be used
We identified three sets of attributes:
1. bio attributes, related to gene information (such as, expression, etc.)
Based on user selection, the node attributes are appended in a single matrix of attributes (`x`)

In the attribute matrix `x` there can be NaN or Infinite values. They are corrected as it follow:
+ NaN is replaced by the mean in the attribute range, 
+ Infinte value is replaced by the maximum in the range.

After Nan and Infinite values fixing, the attributes are normalized with Z-score or MinMax normalization functions.

At the end, only nodes (genes) with E or NE labels are selected for the classification
"""

#@title Choose attributes { form-width: "20%" }
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def impute(x, realcolumns, bincolumns, mode):
  df = x.copy()
  #print(bcolors.OKGREEN + f'- fixing {int(df[realcolumns].isnull().sum().sum())} null values in {len(realcolumns)}/{len(df.columns)} real columns ... ' + bcolors.ENDC, end='')
  p = Stack(bcolors.OKGREEN + f'- fixing {int(df[realcolumns].isnull().sum().sum())} null values in {len(realcolumns)}/{len(df.columns)} real columns ... ' + bcolors.ENDC, max=len(realcolumns))
  if mode == "mean":
      for col in realcolumns: 
        mean_value= df[col].mean()
        if np.isnan(mean_value):
          df = df.drop(columns=[col])
        else:
          df[col].fillna(value=mean_value, inplace=True)
        p.next()
      print(bcolors.OKGREEN + " done!" + bcolors.ENDC,)
  elif mode == "zero":
      for col in df.columns: 
        df[col].fillna(value=0, inplace=True)
      p.next()
      print(bcolors.OKGREEN + " done!" + bcolors.ENDC,)
  else:
    raise Exception(f'Imputation not supported "{mode}"!')
  #print(bcolors.OKGREEN + f'- fixing {int(df[bincolumns].isnull().sum().sum())} null values in {len(bincolumns)}/{len(df.columns)} binary columns' + bcolors.ENDC, end='')
  p = Stack(bcolors.OKGREEN + f'- fixing {int(df[bincolumns].isnull().sum().sum())} null values in {len(bincolumns)}/{len(df.columns)} binary columns ... ' + bcolors.ENDC, max=len(bincolumns))
  for col in bincolumns:
    mostcommon = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
    df[col].fillna(value=mostcommon, inplace=True)
    p.next()
  if len(bincolumns) >  0: print(bcolors.OKGREEN + " done!" + bcolors.ENDC,) 
  print(bcolors.OKGREEN + f'- remaining {df.isnull().sum().sum()} null values' + bcolors.ENDC)
  return df

def normalize(x, columns, mode):
  df = x.copy()
  p = Stack(bcolors.OKGREEN + f'- applying normalization to {len(columns)} columns ... ' + bcolors.ENDC, max=len(columns))
  for col in columns:
    if mode == 'minmax':
        df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())
    elif args.normalize == 'zscore':
        df[col] = (df[col]-df[col].mean())/df[col].std()
    else:
        raise Exception(f'Imputation not supported "{mode}"!')
    p.next()
  print(bcolors.OKGREEN + " done!" + bcolors.ENDC,)
  return df

def load_attributes(attr_file, normalization=None, imputatation=None):
  print(bcolors.HEADER + f'Loading attribute matrix "{attr_file}"...' + bcolors.ENDC)
  x = pd.read_csv(attr_file, index_col=0)
  x = x.select_dtypes(include=np.number)     # drop non numeric attributes
  orignrows = x.shape[0]
  print(bcolors.OKCYAN + f'Statistics...' + bcolors.ENDC)
  print(bcolors.OKGREEN + f'- found {len(x.columns)} data columns' + bcolors.ENDC)
  dup = x.index.duplicated().sum()
  constcolumns = list(x.loc[:, x.apply(pd.Series.nunique)==1].columns)
  nancolumns = x.columns[x.isna().all()].tolist()
  print(bcolors.OKGREEN + f'- {dup} duplicated genes (removed!)' + bcolors.ENDC)
  x = x[~x.index.duplicated(keep='first')]   # remove eventually duplicated index
  if dup > 0:
      x = x[~x.index.duplicated(keep='first')]
  print(bcolors.OKGREEN + f'- {len(nancolumns)} null columns (removed!)' + bcolors.ENDC)
  if len(nancolumns) > 0:
      x = x.drop(nancolumns, axis=1)
  print(bcolors.OKGREEN + f'- {len(constcolumns)} constant columns (removed!)' + bcolors.ENDC)
  if len(constcolumns) > 0:
      x = x.drop(constcolumns, axis=1)
  bincolnames = list(x.loc[:, x.isin([0.0,1.0, np.nan]).all()].columns)
  nbinary = len(bincolnames) if len(bincolnames) > 1 else 0
  print(bcolors.OKGREEN + f'- {nbinary} binary attributes (no normalization!)' + bcolors.ENDC)
  nancount = x.isnull().sum().sum()
  print(bcolors.OKGREEN + f'- {nancount} null values' + bcolors.ENDC)
  ninf = np.isinf(x).values.sum()
  print(bcolors.OKGREEN + f'- {ninf} Infinite values' + bcolors.ENDC)
  realcolumns = list(set(x.columns) - set(bincolnames))
  if imputatation is not None:
    print(bcolors.OKCYAN + f'Imputation with "{args.imputation}"...' + bcolors.ENDC)
    x = impute(x, realcolumns, bincolnames, args.imputation)

  if normalization is not None:
    print(bcolors.OKCYAN + f'Normalization with "{args.normalize}"...' + bcolors.ENDC)
    x = normalize(x, realcolumns, args.normalize) 
    print(bcolors.OKGREEN + f'- skipping {nbinary} binary columns' + bcolors.ENDC)
  return x, orignrows

mergedx = pd.DataFrame()
for attrfile in args.inputfile:
  x, orignrows = load_attributes(attrfile, normalization=args.normalize, imputatation=args.imputation)
  selectedgenes = intersection(x.index.to_list(), selectedgenes)
  print(bcolors.OKCYAN + f'Reduction (to labeled genes)...' + bcolors.ENDC)
  print(bcolors.OKGREEN + f'- dataset reduced from {orignrows} to {len(selectedgenes)} genes' + bcolors.ENDC)
  x = x.loc[selectedgenes]
  mergedx = x if mergedx.empty else pd.concat([mergedx, x], axis=1) 
  print(bcolors.OKGREEN + f'- merged attribute matrix x{mergedx.shape}' + bcolors.ENDC)
x = mergedx
print(bcolors.OKGREEN + f'- final attribute matrix x{x.shape}' + bcolors.ENDC)

# print label distribution
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
df_label_reduced = df_label.loc[selectedgenes]
labels = df_label_reduced[labelname].values
distrib = Counter(labels)
encoder = LabelEncoder()
y = encoder.fit_transform(labels)  
classes_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
rev_classes_mapping = np.array(list(classes_mapping.keys()))

plt.pie(list(distrib.values()), labels=list(distrib.keys()), autopct='%2.1f%%', startangle=90)
distrib = dict(df_label_reduced[labelname].value_counts())
print(bcolors.OKGREEN + f'- new label distribution:  {distrib}' + bcolors.ENDC)

### set balancing in method paramters
classifier_map = {'RF' : 'RandomForestClassifier', 
                  'MLP': 'MLPClassifier', 
                  'SVM' : 'SVC', 
                  'RUS':  'RUSBoostClassifier',
                  'XGB': 'XGBClassifier',
                  'EXT': 'ExtraTreesClassifier',
                  'LGBM': 'LGBMClassifier',
                  'VC' : 'VotingClassifier',
                  'LG': 'LogitBoost'}

classifiers_args = {
    'RF' : {'random_state' : seed, 'class_weight': 'balanced'}, 
    'MLP': {'random_state' : seed}, 
    'SVM' : {'random_state' : seed, 'class_weight': 'balanced'}, 
    'RUS': {'random_state' : seed},
    'LGBM': {'random_state' : seed, 'class_weight': 'balanced'},
    'EXT': {'random_state' : seed, 'class_weight': 'balanced'},
    'XGB': {'random_state' : seed, 'objective' : "binary:logistic", 'eval_metric' : 'logloss', },
    'VC' : {'estimators': [
        ('xt', ExtraTreesClassifier(random_state = seed, class_weight = 'balanced')), 
        ('xgb', XGBClassifier(random_state = seed, objective = "binary:logistic", eval_metric = 'logloss',  scale_pos_weight=0.2)), 
        ('lgb', LGBMClassifier(random_state = seed, class_weight='balanced'))],
         'voting' :'soft'},
    'LG' : {'random_state' : seed, 'n_estimators':200 }
  }
if len(distrib.keys()) == 2:
  first, second = list(distrib.values())
  factor = round(float(first)/float(second), 2) if first < second else round(float(second)/float(first), 2)
  classifiers_args["XGB"] = {'random_state' : seed, 'objective' : "binary:logistic", 'eval_metric' : 'logloss', 'scale_pos_weight' : factor}

## check method compatibility
if x.isnull().sum().sum() > 0 and args.method not in ['LGBM']:
  print(bcolors.FAIL + f'ERR: Method "{args.method}" does not support NaN or Inf input values ... try using -I <imputation-mode>!' + bcolors.ENDC)
  sys.exit(-1)
"""# k-fold cross validation with: SVM, RF, XGB, MLP, RUS

"""

method = args.method #@param ["SVM", "XGB", "RF", "MLP", "RUS", "LGB"]
nfolds = args.folds
genes = x.index.values
X = x.to_numpy()

if args.removefeat:
  print(bcolors.HEADER + f'Selecting features ...' + bcolors.ENDC)
  from sklearn.feature_selection import RFE
  from sklearn.svm import SVR
  estimator = SVR(kernel="linear")
  selector = RFE(estimator, n_features_to_select=100, step=1)
  X = selector.fit_transform(X, y)
  print(bcolors.OKGREEN + f'New attribute matrix x{X.shape}' + bcolors.ENDC)

nclasses = len(classes_mapping)
cma = np.zeros(shape=(nclasses,nclasses), dtype=float)
mm = np.array([], dtype=np.int)
gg = np.array([])
yy = np.array([], dtype=np.int)
accuracies, mccs = [], []
probabilities = np.array([])
predictions = np.array([], dtype=int)
columns_names = ["Accuracy","BA", "Sensitivity", "Specificity","MCC", 'CM']
scores = pd.DataFrame(columns=columns_names)

def do_cv(nfolds, seed):
  predictions = np.array([], dtype=int)
  mm = np.array([], dtype=int)
  gg = np.array([])
  yy = np.array([], dtype=int)
  clf = globals()[classifier_map[method]](**classifiers_args[method])
  scores = pd.DataFrame(columns=columns_names)
  set_seed(seed+1)
  kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
  for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(np.arange(len(X)), y), desc=bcolors.OKGREEN + f'{seed+1}-iter', total=nfolds)):
      train_x, train_y, test_x, test_y = X[train_idx], y[train_idx], X[test_idx], y[test_idx],
      preds = clf.fit(train_x, train_y).predict(test_x)
      cm = confusion_matrix(test_y, preds).astype(float)
      mm = np.concatenate((mm, test_idx))
      yy = np.concatenate((yy, test_y))
      gg = np.concatenate((gg, genes[test_idx]))
      predictions = np.concatenate((predictions, preds))
      scores = pd.concat([scores, pd.DataFrame([[accuracy_score(test_y, preds), balanced_accuracy_score(test_y, preds), 
          cm[0,0]/(cm[0,0]+cm[0,1]), cm[1,1]/(cm[1,0]+cm[1,1]), 
          matthews_corrcoef(test_y, preds), cm]], columns=columns_names, index=[fold])])
  return mm, yy, gg, predictions, scores

print(bcolors.HEADER + f'Classification with method "{method}"...' + bcolors.ENDC)
print(bcolors.OKGREEN + f'- classifier params: {classifiers_args[method]}' + bcolors.ENDC)

if args.jobs == 1:
   print(bcolors.OKCYAN + f'Running seq on 1 cpu...' + bcolors.ENDC)
   out = [do_cv(nfolds, seed) for seed in range(args.repeat)]
else: 
   if args.jobs == -1:
      print(bcolors.OKCYAN + f'Running par on {os.cpu_count()} cpus...' + bcolors.ENDC)
   else:
      print(bcolors.OKCYAN + f'Running par on {args.jobs} cpus...' + bcolors.ENDC)
   out =    Parallel(n_jobs=args.jobs, prefer='threads')(delayed(do_cv)(nfolds, seed) for seed in range(args.repeat))

for iter,res in enumerate(out):
   scores = pd.concat([scores,res[4]])
   cm = confusion_matrix(res[1], res[3]).astype(float)
   cma += cm

cma = cma/float(args.repeat)
dfm_scores = pd.DataFrame(scores.mean(axis=0)).T
dfs_scores = pd.DataFrame(scores.std(axis=0)).T
df_scores = pd.DataFrame([f'{row[0]:.3f}±{row[1]:.3f}' for row in pd.concat([dfm_scores,dfs_scores], axis=0).T.values.tolist()]).T
df_scores.index=[f'{method}']
df_scores['CM'] = [cma]
df_scores.columns = columns_names
print(bcolors.OKGREEN +  tabulate(df_scores, headers='keys', tablefmt='psql') + bcolors.ENDC)
         
