import pandas as pd
import numpy as np
import PreProcessing
import Data_Analysis as DA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import classification_report, accuracy_score
dataset_path = "dataset/spam.csv"

### LOADING THE DATASET ###

# Reading the csv file into a dataframe using pandas
# 'latin-1' encoding helps read the non-ASCII characters
df = pd.read_csv(dataset_path, encoding='latin=1')

# Dropping the unnamed columns (axis = 1 is to specify colums) 
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
# Renaming the columns 
df = df.rename(columns = {"v1" : "label", "v2" : "text"})
# Adding another column for label encoding 
df["label_enc"] = df["label"].map({"ham":0, "spam":1})


### PREPROCESSING THE DATA ###

PreProcessing.MakeLowerCase(df)
# Removing Punctuations 
df['clean_text'] = df['clean_text'].apply(lambda x: PreProcessing.RemovePunctuationHelper(x))
# Removing Stop Words 
df['clean_text'] = df['clean_text'].apply(lambda x: PreProcessing.RemoveStopWordsHelper(x))
# Removing Rare Words
df['clean_text'] = df['clean_text'].apply(lambda x:  PreProcessing.RemoveRareWordsHelper(x, PreProcessing.InFrequencyWords(df)))
# Removing Special Characters 
df['clean_text'] = df['clean_text'].apply(lambda x: PreProcessing.RemoveSpecialCharacterHelper(x))


### DATA ANALYSIS ###

# DA.Plot_Spam_vs_Ham(df)


### TEST TRAIN SPLIT ###
X = np.asanyarray(df["clean_text"])
y = np.asanyarray(df["label_enc"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) 
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


### Multinomial NB ###
tfidf_vec = TfidfVectorizer().fit(X_train)
X_train_vec = tfidf_vec.transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

baseline_model = MultinomialNB() 
baseline_model.fit(X_train_vec,y_train)

nb_accuracy = accuracy_score(y_test, baseline_model.predict(X_test_vec))
print(nb_accuracy)