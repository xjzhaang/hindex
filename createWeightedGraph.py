from tqdm import tqdm
import pandas as pd
import networkx as nx
import numpy as np
import scipy as sp


file = open('author_papers.txt', encoding = 'utf8')
G = nx.read_edgelist('coauthorship.edgelist', delimiter=' ', nodetype=int)
Author_paper_num = {}
Author_paper = {}
for i in tqdm(range(G.number_of_nodes())):
    newline = file.readline()
    author = newline.split(':')[0]
    top5 = newline.split(':')[1].strip().split("-")
    Author_paper[int(author)] = top5

paper_to_author = {}
for k, v in Author_paper.items():
    for x in v:
        if x in paper_to_author:
            paper_to_author[x].append(k)
        else:
            paper_to_author[x] = [k]



nodes = {k: v for v, k in enumerate(list(G.nodes()))}

authors = [author for author in Author_paper.keys()]
papers = [paper for papers in Author_paper.values() for paper in papers]


weighted_adjlist = 0.5 * nx.adjacency_matrix(G).tolil()
for author in tqdm(authors):
    papers = Author_paper[author]
    for p in papers:
        coauthors = paper_to_author[p]
        for coauthor in coauthors:
            if author != coauthor:
                author_node = nodes[author]
                node_author = nodes[coauthor]
                weighted_adjlist[author_node, node_author] += 1

inv_nodes = {v: k for k, v in nodes.items()}
G_weighted = nx.from_scipy_sparse_matrix(weighted_adjlist)
G_weighted = nx.relabel_nodes(G_weighted, inv_nodes)
nx.write_edgelist(G_weighted, 'weighted_coauthorship.edgelist', data=["weight"])
