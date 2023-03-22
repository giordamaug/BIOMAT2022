import argparse
from operator import index
import numpy as np

parser = argparse.ArgumentParser(description='Prova GLEE')
parser.add_argument('-i', "--input", dest='input', metavar='<inputfile>', type=str, help='input ppi file', required=True)
parser.add_argument('-o', "--output", dest='output', metavar='<outputfile>', type=str, help='output embedding file', required=True)
parser.add_argument('-s', "--embedsize", dest='embedsize', metavar='<embedsize>', type=int, help='embed size (default: 128)' , default='128', required=False)
parser.add_argument('-e', "--embedder", dest='embedder', metavar='<embedder>', type=str, help='embedder name (default: RandNE, choice: RandNE|Node2Vec|GLEE|DeepWalk|HOPE|... any other)' , default='RandNE', required=False)
args = parser.parse_args()

from sklearn import preprocessing
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import os,sys

#@title Testo del titolo predefinito { run: "auto", form-width: "20%" }
import pandas as pd
import numpy as np
import networkx as nx
print(f'Loading "{args.input}"...')
df_net = pd.read_csv(args.input, index_col=0)
genes = np.union1d(df_net["source"].values, df_net["target"].values)
idxs = np.arange(len(genes))
gene2idx_mapping = { g : i  for g,i in zip(genes,idxs) }             # create mapping index by gene name
idx2gene_mapping = { i : g  for g,i in zip(genes,idxs) }             # create mapping index by gene name
G = nx.Graph()
G.add_nodes_from(range(len(genes)))                                       # add all nodes (genes, also isolated ones)
if 'weight' in list(df_net.columns):
    edge_list = [(gene2idx_mapping[v[0]], gene2idx_mapping[v[1]], v[2]) for v in list(df_net[['source','target', 'weight']].values)]      # get the edge list (with weights)
    G.add_weighted_edges_from(edge_list)                                      # add all edges
else:
    edge_list = [(gene2idx_mapping[v[0]], gene2idx_mapping[v[1]]) for v in list(df_net[['source','target']].values)]    # get the edge list (with weights)
    G.add_edges_from(edge_list)                                      # add all edges


# GLEE embedding
from karateclub.node_embedding import *
embeddername = "Node2Vec" #@param ["RandNE", "Node2Vec", "GLEE", "DeepWalk"]
print(f'Embedding with method "{embeddername}"...')
embedder = globals()[embeddername](dimensions=args.embedsize)
embedder.fit(G)
embedding = embedder.get_embedding()
embedding_df = pd.DataFrame(embedding, columns = [f'{embeddername}_' + str(i + 1)  for i in range(embedding.shape[1])])
embedding_df['name'] = [idx2gene_mapping[item] for item in embedding_df.index.values]
embedding_df = embedding_df.set_index('name')
embedding_df.to_csv(args.output)
print(embedding_df)

