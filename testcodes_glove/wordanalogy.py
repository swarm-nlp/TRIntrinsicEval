import pickle
import pandas as pd
from gensim.models import KeyedVectors 
from gensim.scripts import glove2word2vec
from gensim.models import KeyedVectors



glove_file = r'glovedefault\vectors.txt'#reading word embeddings
word_vectors = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)#loading vectors

#reading vocabulary list
with open(r'glovedefault\vocablist.txt', 'r',encoding="utf-8") as file:
    words_all=file.read().split()

#create list for best 1, 5 and 10 guesses
top1=[]
top5=[]
top10=[]


def word_analogy_test(word1,word2,word3,target):

    try:
        predicted_word = word_vectors.most_similar(positive=[word2, word3], negative=[word1],topn=10)#vector arithmetic operations
        
    except KeyError:
        print("Bir veya daha fazla kelime vektörü bulunamadı.")
    
    
    result=[t[0] for t in predicted_word]

    if target in result[0]:
        top1.append(target)
    elif target in result[:5]:
        
        top5.append(target)
    elif target in result[:10]:
        
        top10.append(target)
   


# Load the dataset
df = pd.read_csv('msr.csv')

#filtering dataset according to vocabulary and lowering cases
df = df[~df['Manually Corrected Translations'].astype(str).str.contains('\s+', regex=True)]
df['Manually Corrected Translations'] = df['Manually Corrected Translations'].str.lower()
df = df[df['Manually Corrected Translations'].isin(words_all)]

df = df[~df["Unnamed: 7"].astype(str).str.contains('\s+', regex=True)]
df["Unnamed: 7"] = df["Unnamed: 7"].str.lower()
df = df[df["Unnamed: 7"].isin(words_all)]

df = df[~df["Unnamed: 8"].astype(str).str.contains('\s+', regex=True)]
df["Unnamed: 8"] = df["Unnamed: 8"].str.lower()
df = df[df["Unnamed: 8"].isin(words_all)]

df = df[~df["Unnamed: 9"].astype(str).str.contains('\s+', regex=True)]
df["Unnamed: 9"] = df["Unnamed: 9"].str.lower()
df = df[df["Unnamed: 9"].isin(words_all)]


word1list=list(df['Manually Corrected Translations'])
word2list=list(df["Unnamed: 7"])
word3list=list(df["Unnamed: 8"])
targetlist=list(df["Unnamed: 9"])



for word1,word2,word3,target in zip(word1list,word2list,word3list,targetlist):
    
    word_analogy_test(word1,word2,word3,target)
    
#printing results    
print(len(word1list))
print(len(top1))
print(len(top1)/len(word1list))
print(len(top5))
print((len(top1)+len(top5))/len(word1list))
print(len(top10))
print((len(top1)+len(top5)+len(top10))/len(word1list))





