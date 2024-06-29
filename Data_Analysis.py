import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd 

def Plot_Spam_vs_Ham(df): 
    sns.countplot(x=df['label']) 
    plt.show()

