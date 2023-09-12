
![itu-nlp-logo](https://github.com/swarm-nlp/TRIntrinsicEval/assets/72564135/b1a2e1da-45e4-4b29-bd22-e3b5a51791ce)

    
# TRIntrinsicEval
Effective representation of textual data is a prerequisite for most of the downstream tasks, which increases the importance of word embedding evaluation methods. The intrinsic approach assesses the similarity between word representations and human judgements. In this paper, we present a comprehensive intrinsic evaluation of Turkish word embedding models with different tasks using task-specific datasets such as SemEval-2017, MC-30, SimVerb-3500 for word similarity, MSR for word analogy and methods that have not been tested for Turkish before such as concept categorization with BLESS and ESSLLI and outlier detection with 8-8-8 Dataset. While each of these datasets were originally in English, we translated them into Turkish and trained Wor2Vec, FastText and Glove language models with these datasets from scratch. The results suggest that while Word2Vec is generally more successful in word similarity and outlier detection tasks, fastText outperforms other models in word analogy and concept categorization. 


## Models

### FastText
FastText is a word embedding model developed by Facebook for natural language processing (NLP), known for its ability to effectively represent small and rare words using character sequences and subwords. It performs well in various NLP tasks and is accessible through an open-source Python library.

### Word2Vec
Word2vec is a word embedding model widely used in the field of natural language processing (NLP) that represents words as vectors in a continuous vector space, capturing semantic relationships between words. It can be trained using either the "Continuous Bag-of-Words (CBOW)" or "Skip-gram" algorithms, providing meaningful word representations for various NLP tasks.

### Glove
GloVe (Global Vectors for Word Representation) is an unsupervised word embedding model that focuses on learning word vectors based on global co-occurrence statistics from large text corpora. It leverages word co-occurrence frequencies to generate word embeddings that capture semantic relationships and has been widely used in natural language processing tasks.

## Datasets

### Word Similarity
The main idea of this task is to measure the performance of the word representations obtained from the evaluated models by comparing their distance in vector space with the scores obtained from human judgments.

- SemEval-2017
    - Consists of 500 pairs that were assessed for their semantic similarity. The assessment was conducted using a scale ranging from 0 to 4. After translation and
preprocessing steps 319 word pairs remained in the dataset.
- MC-30
    - 30 pairs of words including equal number of word pairs with high, moderate, and low similarity. After translation and preprocessing steps 24 word pairs remained in the dataset.
- SimVerb-3500
    - Includes a total of 3500 verb pairs which are evaluated for their semantic similarity, with ratings ranging from 0 to 10 where lower ratings indicate pairs that are related but not particularly similar. After translation and preprocessing steps 1825 word pairs remained in the dataset.

### Word Analogy
The task of finding a vector that captures the relationship between the words in an analogy is called word analogy. To illustrate, consider a set of words such as
“king”, “queen” and “man” with an unknown fourth word. The aim is to find the missing word that completes the analogy, in this case “woman”, because the relationship is based on gender. For this task, the process involves subtracting the vector representation of “man” from “king” and adding the resulting vector to “queen”. By finding the most similar vector to the resulting vector, we can identify the most appropriate word, which in this case would be “woman”.

- MSR
    - Microsoft Research Syntactic Analogies (MSR) dataset consists of 8,000 questions that are categorized into 16 classes. 4001 word pairs are obtained respectively after translation and preprocessing steps.

### Concept Categorization
The task involves partitioning a given set of words into subsets, where each subset consists of words belonging to distinct categories.

- BLESS
    - Baroni and Lenci Evaluation of Semantic Spaces (BLESS) dataset comprises 200 words from 27 semantic classes. After translation and
preprocessing steps 181 pairs of words left int the dataset.
- ESSLLI
    - European Summer School in Logic, Language and Information dataset consists of 45 words, which are categorized into 9 semantic classes. After translation and
preprocessing steps 32 pairs of words left int the dataset.

### Outlier Detection
As the name indicates, outlier word detection is finding a word that deviates semantically from the rest of a pre-established cluster.

- 8-8-8
    - Contains eight different topics with eight words each and eight outliers.

## Experimentation and Results
![image](https://github.com/swarm-nlp/TRIntrinsicEval/assets/64483224/7285e490-a6f8-4594-b686-6812bd26c279)

## Authors
- [@berfinduman](https://github.com/berfinduman)
- [@oguzaliarslan](https://github.com/oguzaliarslan)
- [@cangunyel](https://github.com/cangunyel)
- [@hakan-erdem](https://github.com/hakan-erdem)
- [@bsnmz](https://github.com/bsnmz)
- [@dgknrsln](https://github.com/dgknrsln)

  
## License
[MIT](https://choosealicense.com/licenses/mit/)

  
