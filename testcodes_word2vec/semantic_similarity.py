import csv
from gensim.models import Word2Vec
import argparse

class ModelSemanticSimilarity:
    def __init__(self, model_paths):
        self.models = {}
        for model_path in model_paths:
            model_name = model_path.split('/')[-1].replace('.bin','') #get the last part of the path as model's name
            self.models[model_name] = Word2Vec.load(model_path)
        self.word_similarities = {}
        
    def change_csv(self, csv_file):
        self.csv_file = csv_file

    def compute_word_similarity(self, word1, word2):
        for model_name, model in self.models.items():
            try:
                similarity = model.wv.similarity(word1, word2)
            except KeyError:
                similarity = None
            self.word_similarities[model_name] = similarity

    def compute_similarities_from_csv(self, result_csv_path):
        print('Starting')
        
        delimiter = ';' if 'mc30' in self.csv_file else ','

        with open(self.csv_file, newline='',  encoding='utf-8') as csvfile,\
                open(result_csv_path, 'w', newline='', encoding='utf-8') as result_csv:
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            fields = ['word1', 'word2'] + list(self.models.keys())

            # normalize the field names 
            norm_field_names = {fieldname.lower(): fieldname for fieldname in reader.fieldnames}

            # check if "similarity" field is present and include in output csv if it is
            if 'similarity' in norm_field_names:
                original_similarity_field_name = norm_field_names['similarity']
                fields.insert(2, original_similarity_field_name)

            writer = csv.DictWriter(result_csv, fieldnames=fields)
            writer.writeheader()

            for row in reader:
                word1 = row['word1'].lower()
                word2 = row['word2'].lower()

                self.compute_word_similarity(word1, word2)

                if not all(value is None for value in self.word_similarities.values()):
                    print(f"Similarities between '{word1}' and '{word2}': {self.word_similarities}")
                    row_data = {'word1': word1, 'word2': word2, **self.word_similarities}

                    # if original similarity is available, add it at the beginning of the output row
                    if original_similarity_field_name is not None: 
                        row_data[original_similarity_field_name] = row[original_similarity_field_name]

                    writer.writerow(row_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Semantic Similarity Arguments')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                        help='List of path/paths to the models')
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                        help='List of path/paths to the dataset')
    args = parser.parse_args()

    model_paths = args.model_paths
    csv_files = args.datasets

    semantic_similarity_calculator = ModelSemanticSimilarity(model_paths)

    for csv_file in csv_files:
        semantic_similarity_calculator.change_csv(csv_file)
        output_file = csv_file.replace('.csv', '_w2v.csv')
        semantic_similarity_calculator.compute_similarities_from_csv(output_file)
