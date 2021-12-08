from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, pdist, squareform
import networkx as nx
import scipy as sp
import os

if not os.path.exists("AbstractEmb"):
    os.makedirs("AbstractEmb")

def create_Embedding(input_file,n_dim,window_size,min_count,epochs,model_name):
    model = Doc2Vec(min_count=min_count,vector_size=n_dim,window=window_size)
    t = TaggedLineDocument(input_file)
    model.build_vocab(t)
    model.train(t,total_examples=model.corpus_count,epochs=epochs)
    model.save("AbstractEmb/"+model_name)


if __name__ == '__main__':

    n_dim = 100
    window_size=5
    input_file = "sentences_line.txt"
    min_count = 2
    epochs = 5
    create_Embedding(input_file,n_dim,window_size,min_count,epochs,model_name)
    print("Model saved")

    
