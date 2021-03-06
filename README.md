# Taco2Vec team submission for kaggle project

<p align="center">
  <img src="https://cdn.paperpile.com/guides/img/h-index-700x350.png" />
</p>

 ## Top 15% in in-class competition at École Polytechnique. Grade: 18/20, A.

# To run the files
First things first, have "abstract.txt", "abstract_paper.txt" and "coauthorship.edgelist" in the directory, then run:

>python inverted_to_normal.py
>
>python remove_stopwords.py

To obtain a txt file "sentences_line.txt" which is "abstract_paper.txt" converted to paragraphs and a "filtered_sentences.txt" which are paragraphs without stopwords.

Then, run
>python createWeightedGraph.py


>python createD2Vmodel.py

To create D2V model, you need to install fastnode2vec, and if you have the latest gensim version, you need to edit the node2vec.py file in your python site-packages/fastnode2vec and change size to vector_size for it to work. 

>python createSimMatrix.py
>

>python createNode2Vec.py

to create the graphs and embeddings.

With gensim=3.8.3 and deepwalk=1.0.3, you can run a commandline to generate deepwalk embeddings, make sure to make the DeepWalkEmbeddings folder:

>deepwalk --format edgelist --input coauthorship.edgelist --number-walks  20 --representation-size 64 --walk-length 20 --window-size 10 --output DeepWalkEmbeddings/coauthor.embeddings

Now with all the data ready, you can load the notebook createData.ipynb to create the X_train, X_test and Y_train csv files. Inside the file you will also find our current best model.

You can also use the notebook learning.ipynb to train the different models.
