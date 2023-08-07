import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import pickle

#loading vocabulary list
inputvector=open("vectorpickle","rb")
vectorvocab=pickle.load(inputvector)

#reading word embeddings
with open('vocablist.txt', 'r',encoding="utf-8") as file:
    words_all=file.read().split()

#loading dataset
df=pd.read_csv("bless.csv",encoding="utf-8")
df=df[["category","wordt"]]

#filtering dataset according to vocabulary and lowering cases
df['wordt'] = df['wordt'].str.strip()
df = df[~df['wordt'].astype(str).str.contains('\s+',regex=True)]
df['wordt'] = df['wordt'].str.lower()
df = df[df['wordt'].isin(words_all)]


#clustering vectors in the space according to category number
def KMeansCluster():
    words=list(df.wordt)
    words=list(set(words))
    vectors={}
    temp_word=list()

    for i in range(len(words)):
        temp_word.append(words[i])
        vectors[words[i]]=vectorvocab.get(words[i])
        print(vectors[words[i]])
    words=temp_word
    kmeans = KMeans(n_clusters=len(df.category.value_counts()))

    X = np.array(list(vectors.values()))
    kmeans.fit(X)

    predictions = kmeans.predict(X)
    word= vectors.keys()
    
    return word,predictions,df



word, prediction,df= KMeansCluster()
data=pd.DataFrame()
data["wordt"]=word
data["prediction"]=prediction

#writing results
data.to_csv("bless-result.csv",encoding="utf-8")
merged_df = pd.merge(df, data, left_on='wordt', right_on='wordt', how='left')
merged_df.to_csv("mergedresult-bless-glove-custom.csv",encoding="utf-8")