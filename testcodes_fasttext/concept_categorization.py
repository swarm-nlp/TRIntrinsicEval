import fasttext
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import sys

def KMeansCluster(opt,path_dataset):
    """Performs K-means clustering on word vectors
    
    opt: Path to the FastText model
    path_dataset: Path to the dataset
    
    Returns: word, predictions, df
    """
    input_file=opt
    print(input_file)
    # Load the FastText model
    model = fasttext.load_model(opt)
    words_all=model.words
    # Read the dataset from CSV file
    df=pd.read_csv(path_dataset,sep=";",encoding="utf-8")
    # Preprocess the dataset
    df=df[["category","word"]]
    df['word'] = df['word'].str.strip()
    df = df[~df['word'].astype(str).str.contains('\s+',regex=True)]
    df['word'] = df['word'].str.lower()
    df = df[df['word'].isin(words_all)]

    words=list(set(list(df.word)))
    vectors={}
    temp_word=list()

    for i in range(len(words)):
        
        temp_word.append(words[i])
        vectors[words[i]]=model.get_word_vector(words[i])

    words=temp_word
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=len(df.category.value_counts()))
    X = np.array(list(vectors.values()))
    kmeans.fit(X)
    predictions = kmeans.predict(X)
    word= vectors.keys()
    return word,predictions,df

def main(path_dataset):

    opts=["model-default.bin","model-cbowdef.bin","model.bin"]

    predictions=[]
    for opt in opts:
        word, prediction,df= KMeansCluster(opt,path_dataset)
        predictions.append(prediction)
    data=pd.DataFrame()
    data["word"]=word

    for i in range(len(opts)):
        data[opts[i]]=predictions[i]
    name_path_dataset=path_dataset
    if "\\" in path_dataset:
        name_path_dataset=path_dataset.split("\\")[-1]
    data.to_csv(f"result-{name_path_dataset}",encoding="utf-8")
    merged_df = pd.merge(df, data, left_on='word', right_on='word', how='left')
    merged_df.to_csv(f"mergedresult-{name_path_dataset}",encoding="utf-8")    

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 1:
        path_dataset = args[0]
        main(path_dataset)

    else:
        sys.exit('''
            Requires:
            path_dataset -> Path of the outlier detection directory
            ''')