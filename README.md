# Taco2Vec team submission for kaggle project

Hi! I'm your first Markdown file in **StackEdit**. If you want to learn about StackEdit, you can read me. If you want to play with Markdown, you can edit me. Once you have finished with me, you can create new files by opening the **file explorer** on the left corner of the navigation bar.


# To run the files
First things first, have "abstract.txt", "abstract_paper.txt" and "coauthorship.edgelist" in the directory, then run:

>python inverted_to_normal.py
>python remove_stopwords.py

To obtain a txt file "sentences_line.txt" which is "abstract_paper.txt" converted to paragraphs and a "filtered_sentences.txt" which are paragraphs without stopwords.

Then, run
>python createD2Vmodel.py
>python createSimMatrix.py
>python createWeightedGraph.py
>python createNode2Vec.py

to create the graphs and embeddings.

With gensim=3.8.3 and deepwalk=1.0.3, you can run a commandline to generate deepwalk embeddings, make sure to make the DeepWalkEmbeddings folder:
>deepwalk --format edgelist --input coauthorship.edgelist --number-walks  20 --representation-size 64 --walk-length 20 --window-size 10 --output DeepWalkEmbeddings/coauthor.embeddings

Now with all the data ready, you can load the notebook createData.ipynb to create the X_train, X_test and Y_train csv files.
Then you can use the notebook learning.ipynb to train the models.
