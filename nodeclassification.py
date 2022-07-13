# -*- coding: utf-8 -*-
"""NodeClassificaytion.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NbmTr3i8M5BIn9UkOuNEY4xP81_EbbZh
"""

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
parser = argparse.ArgumentParser(description='BIOMAT 2022 Workbench')
parser.add_argument('-a', "--attributes", dest='attributes', metavar='<attributes>', nargs="+", default=["BIO", "GTEX", "EMBED"], help='attributes to consider (default BIO GTEX EMBED, values BIO,GTEX,EMBED)', required=False)
parser.add_argument('-c', "--embeddir", dest='embeddir', metavar='<embedding-dir>', type=str, help='embedding directory (default embeddings)', default='embeddings', required=False)
parser.add_argument('-d', "--datadir", dest='datadir', metavar='<data-dir>', type=str, help='data directory (default datasets)', default='datasets', required=False)
parser.add_argument('-l', "--labelname", dest='labelname', metavar='<labelname>', type=str, help='label name (default label_CS_ACH_most_freq)', default='label_CS_ACH_most_freq', required=False)
parser.add_argument('-E', "--essentials", dest='E_class', metavar='<essential-groups>', nargs="+", default=["CS0"], help='CS groups of essential genes (default: CS0, range: [CS0,...,CS9]', required=False)
parser.add_argument('-N', "--nonessenzials", dest='NE_class', metavar='<not-essential-groups>', nargs="+", default=["CS6", "CS7","CS8","CS9"], help='CS groups of non essential genes (default:  CS6 CS7 CS8 CS9], range: [CS0,...,CS9]', required=False)
parser.add_argument('-n', "--network", dest='network', metavar='<network>', type=str, help='network (default: PPI, choice: PPI|MET|MET+PPI)', choices=['PPI', 'MET', 'MET+PPI'], default='PPI', required=False)
parser.add_argument('-Z', "--normalize", dest='normalize', metavar='<normalize>', type=str, help='normalize mode (default: None, choice: None|zscore|minmax)', choices=[None, 'zscore', 'minmax'], default=None, required=False)
parser.add_argument('-e', "--embedder", dest='embedder', metavar='<embedder>', type=str, help='embedder name (default: RandNE, choice: RandNE|Node2Vec|GLEE|DeepWalk|HOPE|... any other)' , default='RandNE', required=False)
parser.add_argument('-m', "--method", dest='method', metavar='<method>', type=str, help='classifier name (default: RF, choice: RF|SVM|XGB|LGBM|MLP)', choices=['RF', 'SVM', 'XGB', 'LGBM', 'MLP'], default='RF', required=False)
parser.add_argument('-V', "--verbose", action='store_true', required=False)
parser.add_argument('-S', "--save-embedding", dest='saveembedding',  action='store_true', required=False)
parser.add_argument('-L', "--load-embedding", dest='loadembedding',  action='store_true', required=False)
parser.add_argument('-X', "--display", action='store_true', required=False)
args = parser.parse_args()

classifier_map = {'RF' : 'RandomForestClassifier', 
                  'MLP': 'MLPClassifier', 
                  'SVM' : 'SVC', 
                  'XGB': 'XGBClassifier',
                  'LGBM': 'LGBMClassifier'}

import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import pandas as pd
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)

"""# Load the network"""

#@title  { run: "auto", form-width: "30%" }
network = args.network #@param ["PPI", "Met", "Met+PPI"]
import sys
if 'google.colab' in sys.modules:
	import tqdm.notebook as tq
else:
	import tqdm as tq
import pandas as pd
import networkx as nx
import os

datapath = args.datadir

"""# Read the labels
Load the label file, select the label type, and print label distribution
"""

from sklearn import preprocessing
from collections import Counter
import matplotlib.pyplot as plt

#@title Testo del titolo predefinito { run: "auto", form-width: "20%" }
labelfile = "multiLabels.csv" #@param {type:"string"}
labelname = args.labelname #@param ["label", "avana0", "avana10", "label_CS_ACH_most_freq"]
import pandas as pd
import numpy as np
label_file = os.path.join(datapath,f'{labelfile}')
print(f'Loading label file "{label_file}"...')
df_label = pd.read_csv(label_file)
df_label['name'] = genes = df_label.index.values                                            # get genes with defined labels (E or NE)
df_label = df_label.reset_index()                                                           # reindex genes by consecutive integers
df_label['index'] = df_label.index
gene2idx_mapping = { v[1] : v[0]  for v in df_label[['index', 'name']].values }             # create mapping index by gene name
idx2gene_mapping = { v[0] : v[1]  for v in df_label[['index', 'name']].values }             # create mapping index by gene name
if labelname == "label_CS_ACH_most_freq":
    #_class, NE_class = ['CS0'], ['CS6', 'CS7', 'CS8', 'CS9']
    new_label_name = 'CS0_vs_CS6-9'
    df_label[new_label_name] = df_label.apply(lambda row: 'E' if row[labelname] in args.E_class \
                                        else 'NE' if row[labelname] in args.NE_class \
                                        else 'ND', axis=1)
    labelname = new_label_name
exclude_labels = ['ND', np.nan]
df_label = df_label[df_label[labelname].isin(exclude_labels) == False]                      # drop any row contaning NaN or SC1-SC5 as value
distrib = Counter(df_label[labelname].values)
selectedgenes = df_label['name'].values
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(df_label[labelname].values)  
classes_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
plt.pie(list(distrib.values()), labels=list(distrib.keys()), autopct='%2.1f%%', startangle=90)
print(bcolors.OKGREEN + f'\t{len(selectedgenes)} labeled genes over a total of {len(genes)}' + bcolors.ENDC)
print(bcolors.OKGREEN + f'\tWorking on label "{labelname}": {classes_mapping} {dict(distrib)}' + bcolors.ENDC)
#plt.show()

"""# Load attributes to be used
We identified three sets of attributes:
1. bio attributes, related to gene information (such as, expression, etc.)
1. GTEX-* attribute, additional biological information of genes 
Based on user selection, the node attributes are appended in a single matrix of attributes (`x`)

In the attribute matrix `x` there can be NaN or Infinite values. They are corrected as it follow:
+ NaN is replaced by the mean in the attribute range, 
+ Infinte value is replaced by the maximum in the range.

After Nan and Infinite values fixing, the attributes are normalized with Z-score or MinMax normalization functions.

At the end, only nodes (genes) with E or NE labels are selected for the classification
"""

#@title Choose attributes { form-width: "20%" }
import re
r = re.compile('^GTEX*')

normalize_node = args.normalize #@param ["", "zscore", "minmax"]
attr_file = os.path.join(datapath,'integratedNet_nodes_bio.csv')
print(f'Loading attribute matrix "{attr_file}"...')
x = pd.read_csv(attr_file)
x = x.drop(columns=['id'])
gtex_attributes = list(filter(r.match, x.columns)) 
bio_attributes = list(set(x.columns).difference(gtex_attributes)) if "BIO" in args.attributes else []
gtex_attributes = gtex_attributes if "GTEX" in args.attributes else [] 
print(bcolors.OKGREEN + f'\tselecting attributes: {args.attributes} for {len(genes)} genes' + bcolors.ENDC)
x = x.filter(items=bio_attributes+gtex_attributes)
print(bcolors.OKGREEN + f'\tfound {x.isnull().sum().sum()} NaN values and {np.isinf(x).values.sum()} Infinite values' + bcolors.ENDC)
for col in x.columns[x.isna().any()].tolist():
  mean_value=x[col].mean()          # Replace NaNs in column with the mean of values in the same column
  if mean_value is not np.nan:
    x[col].fillna(value=mean_value, inplace=True)
  else:                             # otherwise, if the mean is NaN, remove the column
    x = x.drop(col, 1)
if normalize_node == 'minmax':
  print(bcolors.OKGREEN + "\tgene attributes normalization (minmax)..." + bcolors.ENDC)
  x = (x-x.min())/(x.max()-x.min())
elif normalize_node == 'zscore':
  print(bcolors.OKGREEN + "\tgene attributes normalization (zscore)..." + bcolors.ENDC)
  x = (x-x.mean())/x.std()
x = x.loc[selectedgenes]
print(bcolors.OKGREEN + f'\tNew attribute matrix x{x.shape}' + bcolors.ENDC)

"""# Load the PPI+MET network
The PPI networks is loaded from a CSV file, where
*   `from` is the column name for edge source (gene index)
*   `to` is the column name for edge target (gene index)
*   `weight` is the column name for edge weight

"""

if "EMBED" in args.attributes:
  import networkx as nx
  df_net = pd.read_csv(os.path.join(datapath,f'{network.lower()}_edges.csv'), index_col=0)
  print(f'Loading "{network}" network...')
  edge_list = [(gene2idx_mapping[v[0]], gene2idx_mapping[v[1]], v[2]) for v in list(df_net[['source','target', 'weight']].values)]      # get the edge list (with weights)
  if network == "PPI":
    G = nx.Graph()
  else:
    G = nx.DiGraph()
  G.add_nodes_from(range(len(genes)))                                       # add all nodes (genes, also isolated ones)
  G.add_weighted_edges_from(edge_list)                                      # add all edges
  print(bcolors.OKGREEN + "\t" + nx.info(G)  + bcolors.ENDC)
  print(bcolors.OKGREEN + f'\tThere are {len(list(nx.isolates(G)))} isolated genes' + bcolors.ENDC)
  print(bcolors.OKGREEN + f'\tGraph {"is" if nx.is_weighted(G) else "is not"} weighted' + bcolors.ENDC)
  print(bcolors.OKGREEN + f'\tGraph {"is" if nx.is_directed(G) else "is not"} directed' + bcolors.ENDC)

  """# Network embedding with Karateclub""" 

  from karateclub.node_embedding import *
  embeddername = args.embedder #@param ["RandNE", "Node2Vec", "GLEE", "DeepWalk"]
  if not embeddername in dir():
    raise Exception(bcolors.FAIL + f"{embeddername} is not an embedding method supported in karateclub" + bcolors.ENDC)
  print(f'Embedding with method "{embeddername}"...')
  embedfilename = os.path.join(args.embeddir,f'{network}_{embeddername}.csv')
  if args.loadembedding:
    print(bcolors.OKGREEN + f'\tLoading precomputed embedding from file "{embedfilename}"' + bcolors.ENDC)
    embedding_df = pd.read_csv(embedfilename, index_col=0)
  else:
    embedder = globals()[embeddername](dimensions = 128)
    embedder.fit(G)
    embedding = embedder.get_embedding()
    embedding_df = pd.DataFrame(embedding, columns = [f'{embeddername}_' + str(i + 1)  for i in range(embedding.shape[1])])
    embedding_df['name'] = [idx2gene_mapping[item] for item in embedding_df.index.values]
    embedding_df = embedding_df.set_index('name')
  if args.saveembedding:
    embedding_df.to_csv(embedfilename)
    print(bcolors.OKGREEN + f'\Saving embedding to file "{embedfilename}"' + bcolors.ENDC)
  embedding_df = embedding_df.loc[selectedgenes]                                     # keep only embeddings of selected genes (those with labels)
  x = pd.concat([embedding_df, x], axis=1)
  print(bcolors.OKGREEN + f'\tNew attribute matrix x{x.shape}' + bcolors.ENDC)

"""# k-fold cross validation with: SVM, RF, XGB, MLP, RUS

"""

#@title Choose classifier { run: "auto", form-width: "20%" }
method = args.method #@param ["SVM", "XGB", "RF", "MLP", "RUS", "LGB"]
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.neural_network import MLPClassifier
import sys
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from tqdm import tqdm
from sklearn.metrics import *
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier
from tabulate import tabulate
seed=1
set_seed(seed)
nfolds = 5
kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
accuracies, mccs = [], []
X = x.to_numpy()

clf = globals()[classifier_map[args.method]](random_state=seed)
nclasses = len(classes_mapping)
cma = np.zeros(shape=(nclasses,nclasses), dtype=np.int)
mm = np.array([], dtype=np.int)
predictions = np.array([])
columns_names = ["Accuracy","BA", "Sensitivity", "Specificity","MCC", 'CM']
scores = pd.DataFrame(columns=columns_names)
print(f'Classification with method "{method}"...')
#for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(X)), y)):
for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(np.arange(len(X)), y), total=kf.get_n_splits(), desc=bcolors.OKGREEN +  f"{nfolds}-fold")):
    train_x, train_y, test_x, test_y = X[train_idx], y[train_idx], X[test_idx], y[test_idx],
    mm = np.concatenate((mm, test_idx))
    preds = clf.fit(train_x, train_y).predict(test_x)
    cm = confusion_matrix(test_y, preds)
    cma += cm.astype(int)
    predictions = np.concatenate((predictions, preds))
    scores = scores.append(pd.DataFrame([[accuracy_score(test_y, preds), balanced_accuracy_score(test_y, preds), 
        cm[0,0]/(cm[0,0]+cm[0,1]), cm[1,1]/(cm[1,0]+cm[1,1]), 
        matthews_corrcoef(test_y, preds), cm]], columns=columns_names, index=[fold]))
df_scores = pd.DataFrame(scores.mean(axis=0)).T
df_scores.index=[f'{method}']
df_scores['CM'] = [cma]
disp = ConfusionMatrixDisplay(confusion_matrix=cma,display_labels=encoder.inverse_transform(clf.classes_))
disp.plot()
plt.show() if args.display else None
print(bcolors.OKGREEN +  tabulate(df_scores, headers='keys', tablefmt='psql') + bcolors.ENDC)