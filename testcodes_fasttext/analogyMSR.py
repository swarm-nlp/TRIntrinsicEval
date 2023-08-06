import fasttext
import heapq
import pandas as pd 
import sys
def main(opt,path_dataset):
    # Load the FastText model
    model = fasttext.load_model(opt)
    # Read the dataset from CSV file
    df=pd.read_csv(path_dataset,sep=";",encoding="utf-8")
    df=df[["word1", "word2", "word3","target"]]
    print(df.head(),"size:", df.shape)
    # Remove rows with whitespace in any column
    for col in df.columns:
        df = df[~df[col].astype(str).str.contains('\s+',regex=True)]
    words_all=model.words
    vector_list = []
    # Retrieve word vectors
    for word in words_all:
        vector = model.get_word_vector(word)
        vector_list.append(vector)
    # Filter out rows with words that are not present in the word vectors
    for col in df.columns:
        df = df[df[col].isin(words_all)]
    print(df.head(),"size:", df.shape)
    word1=list(df["word1"])
    word2=list(df["word2"])
    word3=list(df["word3"])
    target=list(df["target"])
    result_1=0
    result_5=0
    result_10=0
    # Perform analogy task and evaluate results
    for i in range(len(df)):
        a= model.get_analogies(word2[i], word1[i],word3[i])
        kelimeler = [item[1] for item in a]
        
        if target[i] in kelimeler[:10]:
            result_10+=1
        if target[i] in kelimeler[:5]:
            result_5+=1
        if target[i] == kelimeler[0]:
            print(kelimeler[0], target[i])
            result_1+=1
    
    print("Results", result_1, result_5,result_10)
    print("@10", result_10/ len(df))
    print("@5",result_5/ len(df))
    print("@1",result_1/ len(df))



if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 1:
        path_dataset = args[0]
        path_vectors = args[1]

        main(path_vectors, path_dataset)

    else:
        sys.exit('''
            Requires:
            path_vectors -> Path of the input word vectors
            path_dataset -> Path of the outlier detection directory

            ''')