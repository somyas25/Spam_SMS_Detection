import pandas as pd
import string 

def MakeLowerCase(df):
    df['clean_text'] = df['text'].str.lower()

def RemovePunctuationHelper(text): 
    return text.translate(str.maketrans('', '', string.punctuation))

def RemovePunctuations(df):
    df['clean_text'] = df['clean_text'].apply(lambda x: RemovePunctuationHelper(x))