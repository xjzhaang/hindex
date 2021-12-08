from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, pdist, squareform
import networkx as nx
import scipy as sp

def create_dicts(model,create_sum_dict=True,create_mean_dict=False):
    linecount = open('abstracts.txt')
    N_lines = sum(1 for _ in linecount);linecount.close()

    file = open('abstracts.txt', encoding='utf8')
    paperIds = []
    #get the paper ids
    for i in tqdm(range(N_lines)):
        newLine = file.readline()
        split = newLine.split('----', 1)
        paperIds.append(split[0])

    embedding_id_to_paperids = { i : paperIds[i] for i in range(len(paperIds)) }
    paperids_to_embedding_id = dict((v, k) for k, v in embedding_id_to_paperids.items())

    #mapping authors ids to their top5 papers ids
    file2 = open('author_papers.txt', encoding = 'utf8')
    linecount = open('author_papers.txt', encoding = 'utf8')
    author_line = sum(1 for _ in linecount)
    Auth_dict = {}

    for i in tqdm(range(author_line)):
        newline = file2.readline()
        author = newline.split(':')[0]
        top5 = newline.split(':')[1].strip().split("-")
        Auth_dict[int(author)] = top5
    
    #save the embeddings
    if create_mean_dict:
        mean_dict = {}
    if create_sum_dict:
        sum_dict = {}
    papers = set()
    for key in tqdm(Auth_dict.keys()):
        list_id = [i for i in Auth_dict[key]]
        
        emb = []
        for i in list_id:
            papers.add(i)
            try:
                code = paperids_to_embedding_id[i]
                emb.append(model.dv[code])
            except:
                #since authors have papers that are missing from the abstract.txt
                pass
        if create_mean_dict:
            mean = np.mean(emb,axis=0)
            mean_dict[key] = mean
        if create_sum_dict:
            sum_ = np.sum(emb,axis=0)
            sum_dict[key] = sum_

    if create_mean_dict:
        return mean_dict
    if create_sum_dict:
        return sum_dict
    if create_mean_dict and create_sum_dict:
        return mean_dict,sum_dict

def create_similarity_matrix(model):
    sum_dict = create_dicts(model,filename,input_file)
    df = pd.DataFrame.from_dict(sum_dict)
    G = nx.read_edgelist(input_file,delimiter=' ', nodetype=int)
    nodes = {k: v for v, k in enumerate(list(G.nodes()))}

    n = len(G.nodes())
    matrix = sp.sparse.csr_matrix((n, n)).tolil()
    for author_id1 in tqdm(df):
        curr_emb = df[author_id1].to_numpy()
        curr_author = nodes[author_id1]
        for author_id2 in G.neighbors(author_id1):
                matrix[curr_author, nodes[author_id2]] = np.dot(curr_emb, df[author_id2].to_numpy())

    inv_nodes = {v: k for k, v in nodes.items()}
    SG = nx.from_scipy_sparse_matrix(matrix)
    SG = nx.relabel_nodes(SG, inv_nodes)
    nx.write_multiline_adjlist(SG,filename)


if __name__ == '__main__':
    model = Doc2Vec.load("AbstractEmb/"+model_name)
    input_file = 'coauthorship.edgelist'
    filename = 'sum_sim_authors.adjlist'
    print("Creating the similarity matrix and storing it in: ","SimilarityMatrices"+filename,input_file)
    create_similarity_matrix(model)
    print("Done.")

    