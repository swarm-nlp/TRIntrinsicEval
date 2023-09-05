
![itu-nlp-logo](https://github.com/swarm-nlp/TRIntrinsicEval/assets/72564135/b1a2e1da-45e4-4b29-bd22-e3b5a51791ce)

    
# TRIntrinsicEval
Effective representation of textual data is a prerequisite for most of the downstream tasks, which increases the importance of word embedding evaluation methods. The intrinsic approach assesses the similarity between word representations and human judgements. In this paper, we present a comprehensive intrinsic evaluation of Turkish word embedding models with different tasks using task-specific datasets such as SemEval-2017, MC-30, SimVerb-3500 for word similarity, MSR for word analogy and methods that have not been tested for Turkish before such as concept categorization with BLESS and ESSLLI and outlier detection with 8-8-8 Dataset. While each of these datasets were originally in English, we translated them into Turkish and trained Wor2Vec, FastText and Glove language models with these datasets from scratch. The results suggest that while Word2Vec is generally more successful in word similarity and outlier detection tasks, fastText outperforms other models in word analogy and concept categorization. 


## Models

### FastText
FastText is a word embedding model widely used in the field of natural language processing (NLP). Developed by Facebook, this model provides better results than other word embedding models, especially in representing small and rare words. The FastText model uses the character sequence and subwords of a word to represent it. This way, it can better capture the semantic structure of the word. For example, the word "curious" is broken down into the subwords "curi" and "ous", and the vectors of these subwords contribute to the vector of the word. The FastText model uses either the "Continuous Bag-of-Words (CBOW)" or "Skip-gram" algorithms for word embedding. CBOW is based on predicting a word from the context of other words, while Skip-gram is used to predict other words in the given context. When used for word embedding, the FastText model shows high performance in NLP tasks such as word similarity, word classification, sentiment analysis, and tagging. Additionally, the FastText model is also used for developing low-resource language models in NLP. FastText is an open-source software, and there is a Python library developed by Facebook for using the FastText model. As a result, researchers and developers can easily use and apply the FastText model in their NLP applications.

### Word2Vec
Word2vec is a word embedding model widely used in the field of natural language processing (NLP). This model represents a word according to its coordinates in vector space, allowing for mathematical expression of similarities and differences in word meanings. The Word2vec model works with two different algorithms: Continuous Bag-of-Words (CBOW) and Skip-gram. CBOW is based on predicting a word from other words in a given context, while Skip-gram is used to predict other words in a given context from a word. These algorithms are trained using a large text corpus, and a vector is created for each word. The Word2vec model performs well in NLP tasks such as word similarity, word classification, sentiment analysis, and tagging. Additionally, this model requires less processing power compared to calculating high-dimensional features by using low-dimensional vector representations. The Word2vec model is open-source software and can be used in many programming languages such as Python, Java, and C++.

### Glove
The GloVe (Global Vectors) model is a machine learning algorithm used in the field of natural language processing (NLP). GloVe is one of the most commonly used word embedding methods, which is used to represent words as numerical vectors. Words differ in their coordinate positions in vector space. In this way, word vectors allow mathematical expression of similarities and differences in word meanings. The GloVe model uses co-occurrence statistics between words to learn word vectors. The model calculates the word co-occurrence matrix from a large text corpus and discovers patterns in this matrix. Then, it creates a vector for each word using these patterns. The GloVe model performs well in NLP tasks such as word similarity, word relationships, and word meanings. In particular, the GloVe model is highly effective for NLP applications that require finding word similarity or relationships. The GloVe model was developed at Stanford University and is available as open-source code. Therefore, researchers and developers can easily use and apply the GloVe model in natural language processing applications.


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

  
