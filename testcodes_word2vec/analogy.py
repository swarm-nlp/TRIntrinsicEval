import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm
import csv
import argparse

class Word_Analogy():
    def __init__(self, data: pd.DataFrame, model_paths: list, model_names:list, path:str, number:int) -> None:
        '''
        data: Dataframe containing the words to be compared
        model_paths: List of paths to the models
        model_names: List of names of the models
        csv_path: Path to the output csv file
        txt_path: Path to the output txt file
        number: Calculates the number of words to be compared

        return: None
        '''
        self.data = data
        self.models = self.Model_Loader(model_paths)
        self.model_names = model_names
        self.csv_path = path + '.csv'
        self.txt_path = path + '.txt'
        self.number = number
        self.calculate = self.Analogy_Calculator()

    def Model_Loader(self, model_paths: str) -> None:
        '''
        Load the models
        '''
        models = [Word2Vec.load(model_path) for model_path in model_paths]

        return models

        
    def word_analogy(self, word1, word2, word3, model):
        ''' 
        x_1 is to y_1 as x_2 is to y_2
        word1: y_1
        word2: x_1
        word3: x_2
        model: Word2Vec model

        return: Result  
        '''
        result = model.wv.most_similar(positive=[word1, word3], negative=[word2])
        return result 


    def Analogy_Calculator(self) -> None:    
        print('Starting')
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['word1', 'word2', 'word3', 'target'] + self.model_names)

            accuracies = {model_name: 0 for model_name in self.model_names}

            total_examples = 0

            for index, row in tqdm(data.iterrows(), total=5000):
                try:
                    word1 = row['word1'].lower()
                    word2 = row['word2'].lower()
                    word3 = row['word3'].lower()
                    target = row['target'].lower()

                    predicted_words = []

                    for model, model_name in zip(self.models, self.model_names):
                        similar_words = [word for word, _ in self.word_analogy(word2, word1, word3, model)[:self.number]]
                        predicted_words.append(similar_words)

                        if target in similar_words:
                            accuracies[model_name] += 1

                    writer.writerow([word1, word2, word3, target] + predicted_words)

                    total_examples += 1

                except KeyError:
                    continue

            print('Finished')
            with open(self.txt_path, 'w') as f:
                f.write("Model Accuracies:\n")
                for model_name, accuracy in accuracies.items():
                    model_accuracy = accuracy / total_examples
                    f.write(f"{model_name}: {model_accuracy * 100:.2f}%\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Word Analogy Arguments')
    parser.add_argument('--data', type=str, required=True, help='Path to the data file')
    parser.add_argument('--model_path', type=str, nargs='+', required=True, help='List of path/paths to the models')
    parser.add_argument('--number', type=int, required=True, help='Number of words to be compared')

    args = parser.parse_args()
    data = pd.read_csv(args.data)
    model_paths = args.model_path

    #get the last part of the path as model's name
    model_names = [model_path.split('/')[-1].replace('.bin','') for model_path in model_paths]


    for number in [args.number]:
        path = f'first-{number}-words'
        Word_Analogy(data, model_paths, model_names, path, number)
