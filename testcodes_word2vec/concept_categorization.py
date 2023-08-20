import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from gensim.models import Word2Vec

def KMeansCluster(opt, path_dataset):
    """Performs K-means clustering on word vectors
    
    opt: Path to the Word2Vec model
    path_dataset: Path to the dataset
    
    Returns: word, predictions, df
    """
    input_file = opt
    print(input_file)
    # Load the Word2Vec model
    model = Word2Vec.load(opt)
    words_all = model.wv.index_to_key
    # Read the dataset from CSV file
    df = pd.read_csv(path_dataset, encoding="utf-8")
    # Preprocess the dataset, not mine very weird operations tbf
    df.rename(columns={'category':'label', 'word':'text'}, inplace=True)
    df['text'] = df['text'].str.strip()
    df = df[~df['text'].astype(str).str.contains('\s+', regex=True)]
    df['text'] = df['text'].str.lower()
    df = df[df['text'].isin(words_all)]

    words = list(set(list(df.text)))
    vectors = {}
    temp_word = list()

    for i in range(len(words)):
        temp_word.append(words[i])
        vectors[words[i]] = model.wv[words[i]]

    words = temp_word
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=len(df.label.value_counts()), n_init='auto')
    X = np.array(list(vectors.values()))
    kmeans.fit(X)
    predictions = kmeans.predict(X)
    word = vectors.keys()
    return word, predictions, df

def main(path_dataset, models):

    opts = models

    predictions = []
    for opt in opts:
        word, prediction, df = KMeansCluster(opt, path_dataset)
        predictions.append(prediction)

    data = pd.DataFrame()
    data["word"] = word

    for i in range(len(opts)):
        data[opts[i]] = predictions[i]

    name = path_dataset.split('/')[-1]
    merged_df = pd.merge(df, data, left_on='text', right_on='word', how='left')
    merged_df.drop(columns=['text'], inplace=True)
    merged_df.to_csv(f"{name}_merged_concept.csv", encoding="utf-8")    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform K-means clustering on word vectors')
    parser.add_argument('--path_dataset', type=str, help='Path to the dataset')
    parser.add_argument('--models', type=str, nargs='+', help='Path(s) to the Word2Vec model(s)')
    args = parser.parse_args()

    path_dataset = args.path_dataset
    models = args.models
    main(path_dataset, models)
