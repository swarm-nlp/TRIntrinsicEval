import logging
import multiprocessing
from tqdm import tqdm
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class My_Word2Vec():
    def __init__(self, file_path:str, vector_size:int=100, window:int=5, min_count:int=5, epoch:int=5) -> None:
        '''
        file_path: Path to the file
        vector_size: Dimensionality of the word vectors (default=100)
        window: Maximum distance between the current and predicted word within a sentence (default=5)
        min_count: Ignores all words with total frequency lower than this (default=5)
        epoch: Number of iterations (default=5)

        return: Word2Vec model
        '''
        self.file_path = file_path
        self.chunk_size = 10000
        # Parameters of model
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = multiprocessing.cpu_count()
        self.epochs = epoch
        # Sentences
        self.sentences = list(self.read_text_file(self.file_path, self.chunk_size))

        self.model = self.train_word2vec_model(self.sentences)

    # Read the txt file
    def read_text_file(self, filename:str, chunk_size:int):
        size = self.file_size(filename)

        with open(filename, 'r', encoding='utf-8') as file:
            bytes_read = 0
            pbar = tqdm(total=size, unit='B', unit_scale=True)

            while bytes_read < size:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                bytes_read += len(chunk)
                pbar.update(len(chunk))
                yield chunk.lower().split()
            pbar.close()

    # Get file size in bytes
    def file_size(self, filename):
        with open(filename, 'rb') as file:
            file.seek(0, 2)
            size = file.tell()
        return size

    # Train Word2Vec model
    def train_word2vec_model(self, input):
        sentences = list(input)
        with tqdm(total=len(sentences), unit=' sentences') as pbar:
            #model = Word2Vec(sentences=sentences, vector_size=self.vector_size, window=self.window,min_count=self.min_count, workers=self.workers, epochs=self.epochs)
            #model = Word2Vec(sentences=sentences, sg=1)
            model = Word2Vec(sentences=sentences, vector_size=self.vector_size, 
                             window=self.window,min_count=self.min_count, workers=self.workers, epochs=self.epochs, sg=1)

            pbar.update(len(sentences))

        return model

input_address = 'data/preprocessed.txt'

# Train Word2Vec model
# Former is my settings, latter is default settings
model = My_Word2Vec(input_address, vector_size=300, window=8, min_count=5, epoch=10).model
#model = My_Word2Vec(input_address).model

# Save model
#model.save('word2vec_model/my_settings/word2vec_model.bin')
model.save('word2vec_model/skipgram_my_settings/my_skipgram.bin')

# Save word vectors
with open("word2vec_model/skipgram_my_settings/my_skipgram_vectors.txt", "w", encoding="utf-8") as f:
    for word in tqdm(model.wv.key_to_index):
        vector = " ".join(str(x) for x in model.wv[word])
        f.write(word + " " + vector + "\n")