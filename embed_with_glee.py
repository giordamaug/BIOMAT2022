import argparse
from operator import index
import numpy as np

parser = argparse.ArgumentParser(description='Prova GLEE')
parser.add_argument('-d', "--datadir", dest='datadir', metavar='<data-dir>', type=str, help='data directory (default datasets)', default='datasets', required=False)
parser.add_argument('-c', "--embeddir", dest='embeddir', metavar='<embedding-dir>', type=str, help='embedding directory (default embeddings)', default='embeddings', required=False)
parser.add_argument('-n', "--network", dest='network', metavar='<network>', type=str, help='network (default: PPI, choice: PPI|MET|MET+PPI)', choices=['PPI', 'MET', 'MET+PPI'], default='PPI', required=False)
parser.add_argument('-l', "--labelname", dest='labelname', metavar='<labelname>', type=str, help='label name (default label_CS_ACH_most_freq)', default='label_CS_ACH_most_freq', required=False)
parser.add_argument('-F', "--labelfile", dest='labelfile', metavar='<labelfile>', type=str, help='label filename (default multiLabels.csv)', default='multiLabels.csv', required=False)
parser.add_argument('-P', "--netfile", dest='netfile', metavar='<netfile>', type=str, help='network filename (default ppi_edges.csv)', default='ppi_edges.csv', required=False)
parser.add_argument('-S', "--save-embedding", dest='saveembedding',  action='store_true', required=False)
parser.add_argument('-L', "--load-embedding", dest='loadembedding',  action='store_true', required=False)
parser.add_argument('-s', "--embedsize", dest='embedsize', metavar='<embedsize>', type=int, help='embed size (default: 128)' , default='128', required=False)
parser.add_argument('-e', "--embedder", dest='embedder', metavar='<embedder>', type=str, help='embedder name (default: RandNE, choice: RandNE|Node2Vec|GLEE|DeepWalk|HOPE|... any other)' , default='RandNE', required=False)
args = parser.parse_args()

datapath = args.datadir
network = args.network

from sklearn import preprocessing
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import os,sys

#@title Testo del titolo predefinito { run: "auto", form-width: "20%" }
labelfile = args.labelfile #@param {type:"string"}
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

import networkx as nx
netfile = os.path.join(datapath,args.netfile)
print(f'Loading "{netfile}"...')
df_net = pd.read_csv(netfile, index_col=0)
if network == "PPI":
    G = nx.Graph()
else:
    G = nx.DiGraph()
G.add_nodes_from(range(len(genes)))                                       # add all nodes (genes, also isolated ones)
if 'weight' in list(df_net.columns):
    edge_list = [(gene2idx_mapping[v[0]], gene2idx_mapping[v[1]], v[2]) for v in list(df_net[['source','target', 'weight']].values)]      # get the edge list (with weights)
    G.add_weighted_edges_from(edge_list)                                      # add all edges
else:
    edge_list = [(gene2idx_mapping[v[0]], gene2idx_mapping[v[1]]) for v in list(df_net[['source','target']].values)]    # get the edge list (with weights)
    G.add_edges_from(edge_list)                                      # add all edges


# GLEE embedding
from glee import eigenmaps
embeddername = args.embedder #@param ["RandNE", "Node2Vec", "GLEE", "DeepWalk"]
print(f'Embedding with method "{embeddername}"...')
embedfilename = os.path.join(args.embeddir,f'{network}_{embeddername}_{args.embedsize}.csv')
embedding = eigenmaps(G, args.embedsize, method='glee')
embedding_df = pd.DataFrame(embedding, columns = [f'{embeddername}_' + str(i + 1)  for i in range(embedding.shape[1])])
embedding_df['name'] = [idx2gene_mapping[item] for item in embedding_df.index.values]
embedding_df = embedding_df.set_index('name')
if args.saveembedding:
    embedding_df.to_csv(embedfilename)
print(embedding_df)

