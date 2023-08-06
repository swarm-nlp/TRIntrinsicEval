import fasttext
import numpy as np
import pandas as pd
import sys

def cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity of two given vectors
    vec1:vector embedding of the word1
    vec2:vector embedding of the word2

    Return: similarity
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def main(path_dataset, path_vectors):
    # Load the FastText model
    model = fasttext.load_model(path_vectors)
    words_all= model.words

    lines= []
    print(path_dataset)
    # Read the dataset from CSV file

    df=pd.read_csv(path_dataset,sep=",",encoding= 'utf-8')
    words=list(zip(df['word1'], df['word2']))
    # Preprocess the dataset
    df = df[~df['word1'].str.contains('\s+', regex=True)]
    df = df[~df['word2'].str.contains('\s+', regex=True)]
    df['word1'] = df['word1'].str.lower()
    df['word2'] = df['word2'].str.lower()
    df = df[df['word1'].isin(words_all)]
    df = df[df['word2'].isin(words_all)]
    words=list(zip(df['word1'], df['word2']))


    similarities=[]
    for i in words:
        # Get vector embeddings for each word pair
        vec1 = model.get_word_vector(i[0])
        vec2 = model.get_word_vector(i[1])
        similarity = cosine_similarity(vec1, vec2)
        similarities.append(similarity)
        print(i[0]," " , i[1],": ", f"Benzerlik: {similarity}")
    
    df["similarity-fasttext-result"]=similarities
    df.to_csv(f"similarities{path_dataset}",sep=";",encoding="utf-8")

if __name__ == '__main__':

    args = sys.argv[1:]

    if len(args) == 2:

        path_dataset = args[0]
        path_vectors = args[1]
        main(path_dataset, path_vectors)

    else:
        sys.exit('''
            Requires:
            path_dataset -> Path of the outlier detection directory
            path_vectors -> Path of the input word vectors
            ''')