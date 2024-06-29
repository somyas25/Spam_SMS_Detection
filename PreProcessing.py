import pandas as pd
import string 
from nltk.corpus import stopwords
from collections import Counter 
import re

def MakeLowerCase(df):
    df['clean_text'] = df['text'].str.lower()

def RemovePunctuationHelper(text): 
    return text.translate(str.maketrans('', '', string.punctuation))

def RemoveStopWordsHelper(text):
    stop_words_set = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word not in stop_words_set])

def InFrequencyWords(df):
    word_count = Counter()
    for text in df['clean_text']:
        for word in text.split():
            word_count[word] += 1
    return set(word for (word,wc) in word_count.most_common()[:-10:-1])
    
def RemoveRareWordsHelper(text, infreq):
    return " ".join([word for word in text.split() if word not in infreq])

def RemoveSpecialCharacterHelper(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text
