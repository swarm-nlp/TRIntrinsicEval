import numpy as np
import pandas as pd
import pickle

#loading vocabulary list
inputvector=open("vectorpickle","rb")
vectorvocab=pickle.load(inputvector)

#reading word embeddings
with open('vocablist.txt', 'r',encoding="utf-8") as file:
    words_all=file.read().split()

#loading dataset
df=pd.read_csv("semeval.csv",encoding="utf-8")

words=list(zip(df['wordt1'], df['wordt2']))

#filtering dataset according to vocabulary and lowering cases
df = df[~df['wordt1'].astype(str).str.contains('\s+', regex=True)]
df = df[~df['wordt2'].astype(str).str.contains('\s+', regex=True)]
df['wordt1'] = df['wordt1'].str.lower()
df['wordt2'] = df['wordt2'].str.lower()
df = df[df['wordt1'].isin(words_all)]
df = df[df['wordt2'].isin(words_all)]
words=list(zip(df['wordt1'], df['wordt2']))

#calculating word pairs similarity based on the cosine of the angle between them
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm1 * norm2)
    return similarity
similarities=[]
for i in words:
    vec1 = vectorvocab.get(i[0])
    vec2 = vectorvocab.get(i[1])
    similarity = cosine_similarity(vec1, vec2)
    similarities.append(similarity)
    print(i[0]," " , i[1],": ", f"Benzerlik: {similarity}")
    

#writing results
dfs= pd.DataFrame(similarities)    
dfs.to_csv('outputsim.csv', index=False)    