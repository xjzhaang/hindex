import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# f = open("sentences.txt", "w", encoding="utf-8")
fline = open("sentences_line.txt", "w", encoding="utf-8")

linecount = open('abstracts.txt')
line = sum(1 for _ in linecount)
file = open('abstracts.txt', encoding='utf8')
stopwords_ = set(stopwords.words('english'))
Absdict = {}
for i in tqdm(range(line)):
    newLine = file.readline()
    split = newLine.split('----', 1)
    Dict = eval(split[1])  
    #print(Dict)
    newDict = {}
    for key, value in Dict['InvertedIndex'].items():
        for new_key in value:
            newDict[new_key] = key
    #f.write(newLine)
    s = ""
    for i in range(Dict['IndexLength']):
        try:
            # word = newDict[i].strip()
            word = newDict[i].replace("[", "").replace(
        "]", "").replace("\n", "").replace("\'", "").replace("\"", "").replace("\r","")
            s += word + " "
        except:
            pass
    s+='\n'
    fline.write(s)
    
# f.close()
fline.close()