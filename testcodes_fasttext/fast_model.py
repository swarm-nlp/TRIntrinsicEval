import gzip
import pandas as pd
import fasttext
import numpy as np
import sys
def main(train_data_path):
    # Train the FastText model
    model = fasttext.train_unsupervised(input=filter_train_data, model="cbow" ,lr=0.1, epoch=10, wordNgrams=2,minCount=5)
    # Save the trained model
    model.save_model("model-cbow.txt")
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 1:
        filter_train_data = args[0]
    else:
        sys.exit('''
            Requires:
            filter_train_data -> Path of the wiki dataset.
            ''')