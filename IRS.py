#Import Files
import nltk
import string
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
from collections import Counter
from wordcloud import WordCloud

#Open and Read document
stop_words = set(stopwords.words('english'))
doc_file = open(r'C:\Home Folder\Rohan\Nirma\Semester 7\IRS\Labs\P3_15BIT049_Document.txt', 'rt')
doc1 = doc_file.read()
doc_file.close()

#Preprocessing
#Convert to Lower Case
doc1 = doc1.lower()

#Removes Punctuations
translation_table = str.maketrans("","",string.punctuation)
#Remove \n for new line to space
doc1.replace('\n', ' ')
doc1 = doc1.translate(translation_table)

#Tokenizer
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(doc1)
cloud_input = ''

#Stopwords Removal
without_stopwords = []
for w in tokens:
    if w not in stop_words:
        without_stopwords.append(w)
ps = PorterStemmer()

#Stemming
stemmed_words = []
for w in without_stopwords:
    stemmed_words.append(ps.stem(w))

#Frequency Creator
counts = Counter(stemmed_words)
l , v = zip(*counts.items())
l = np.array(l)
v = np.array(v)

ind = np.arange(len(l))

bar_width = 1.0

#Histogram
plt.bar(ind, v)
plt.xticks(ind, l, rotation='vertical')
plt.show()

#Word_Cloud
for w in stemmed_words:
    cloud_input = cloud_input + w + ' '
wc = WordCloud(width = 800, height = 800, min_font_size=10).generate(cloud_input)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()
