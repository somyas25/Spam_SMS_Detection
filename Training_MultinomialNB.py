import pandas as pd
import PreProcessing
dataset_path = "dataset/spam.csv"

# Reading the csv file into a dataframe using pandas
# 'latin-1' encoding helps read the non-ASCII characters
df = pd.read_csv(dataset_path, encoding='latin=1')

# Dropping the unnamed columns (axis = 1 is to specify colums) 
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
# Renaming the columns 
df = df.rename(columns = {"v1" : "label", "v2" : "text"})
# Adding another column for label encoding 
df["label_enc"] = df["label"].map({"ham":0, "spam":1})

PreProcessing.MakeLowerCase(df)
PreProcessing.RemovePunctuations(df)
PreProcessing.RemoveStopWords(df)
print(df.head())

