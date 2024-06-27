import pandas as pd
import string 
from nltk.corpus import stopwords

def MakeLowerCase(df):
    df['clean_text'] = df['text'].str.lower()

def RemovePunctuationHelper(text): 
    return text.translate(str.maketrans('', '', string.punctuation))

def RemovePunctuations(df):
    df['clean_text'] = df['clean_text'].apply(lambda x: RemovePunctuationHelper(x))

def RemoveStopWordsHelper(text):
    stop_words_set = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word not in stop_words_set])

def RemoveStopWords(df):
    df['clean_text'] = df['clean_text'].apply(lambda x: RemoveStopWordsHelper(x))