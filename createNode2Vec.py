from fastnode2vec import Graph, Node2Vec
import pandas as pd
import numpy as np
import networkx as nx
import os

if not os.path.exists("Node2VecEmb"):
    os.makedirs("Node2VecEmb")

G_weighted = nx.read_edgelist("weighted_coauthorship.edgelist", nodetype=int, data=(("weight", float),))
G_sim = nx.read_multiline_adjlist("sum_sim_authors.adjlist", nodetype=int)


edges_arr = [(str(edge[0]), str(edge[1]), G_weighted[edge[0]][edge[1]]["weight"]) for edge in G_weighted.edges]
node2vec = Node2Vec(Graph(edges_arr, directed=False, weighted=True), dim=50, walk_length=20, context=10, p=1, q=0.5, workers=12)
node2vec.train(epochs=100)
node2vec.wv.save_word2vec_format("Node2VecEmb/n2v_g_weighted.nodevectors")

edges_arr = [(str(edge[0]), str(edge[1]), G_sim[edge[0]][edge[1]]["weight"]) for edge in G_sim.edges]
node2vec = Node2Vec(Graph(edges_arr, directed=False, weighted=True), dim=50, walk_length=20, context=10, p=1, q=0.5, workers=12)
node2vec.train(epochs=100)
node2vec.wv.save_word2vec_format("Node2VecEmb/n2v_g_sim.nodevectors")
