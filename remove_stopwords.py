from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
stopwords_set = set(stopwords.words('english'))

with open("sentences_line.txt") as doc:
    words = doc.read()
    with open("filtered_sentences.txt", 'w') as set_doc:
        for word in tqdm(words.split(' ')):
            if not word in stopwords_set:
                set_doc.write(word + ' ')