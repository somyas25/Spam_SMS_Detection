import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# loading the csv file with latin-1 encoding so that it can read the non-ASCII characters
df = pd.read_csv("/Users/somyasrivastava/Desktop/Projects/Spam_SMS_Detection/dataset/spam.csv" ,encoding='latin-1') 
# printing the first few columns of the dataframe
# print(df.head())

# dropping the unnamed columns (axis = 1) 
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
# renaming the columns 
df = df.rename(columns = {"v1" : "Label", "v2" : "Text"})
# adding another column for label encoding 
df["Label_Enc"] = df["Label"].map({"ham":0, "spam":1})
#print(df.head())

# making a plof of the number of spam and ham messages 
sns.countplot(x=df['Label']) 
plt.show()
