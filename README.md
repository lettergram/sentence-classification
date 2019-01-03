Sentance Classifcation
======================

The goal of this project is to classify sentences, based on type:

- Statement (Declarative Sentence)
- Question (Interrogative Sentence)
- Exclamation (Exclamatory Sentence)
- Command (Imperative Sentence)

Each of the above broad sentence categories can be expanded and can be made more indepth. The way these networks and scripts are designed it should be possible expand to classify other sentence types, provided the data is provided.

This was developed for applications at [Metacortex](https://metacortex.me) and is accompanied by a guide on building practical/applied neural networks on [austingwalters.com](https://austingwalters.com).

Please, feel free to add PRs to update, improve, and use freely!

---------------------

## Supplemental Material

This repository was created in conjunction with a guide titled[]Nueral Networks to Production, From an Engineer](https://austingwalters.com/neural-networks-to-production-from-an-engineer/).

Below is the guides table of contents:

* [Acquiring & formatting data for deep learning applications](https://austingwalters.com/data-acquisition-and-formatting-for-deep-learning-applications/)
* [Word embedding and data splitting](https://austingwalters.com/word-embedding-and-data-splitting/)
* [Bag-of-words to classify sentence types (Dictionary)](https://austingwalters.com/bag-of-words-to-classify-sentence-types/)
* [Classify sentences via a multilayer perceptron (MLP)](https://austingwalters.com/classify-sentences-via-a-multilayer-perceptron-mlp/)
* [Classify sentences via a recurrent neural network (LSTM)](https://austingwalters.com/classify-sentences-via-a-recurrent-neural-network-lstm/)
* [Convolutional neural networks to classify sentences (CNN)](https://austingwalters.com/convolutional-neural-networks-cnn-to-classify-sentences/)
* [FastText for sentence classification (FastText)](https://austingwalters.com/fasttext-for-sentence-classification/)
* [Hyperparameter tuning for sentence classification](https://austingwalters.com/hyperparameter-tuning-for-sentence-classification/)


---------------------

## Dataset

The dataset is created from parsing out the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset and combining it with the [SPAADIA](http://martinweisser.org/index.html#Amex_a) dataset. 

The samples in the dataset:

* Command 1672
* Statement 81065
* Question 131219

Note: Questions in this case are only one sentence, statements are a single sentence or more. They are classified correctly, but don't include sentences prior to questions. 

## Results

With the above, we are able to get the following accuracy:

| Model | Accuracy | Train Speed | Classification Speed |
| -------- | ----------- | --------- | ----------------- |
| Dict | 85% | **Fastest** | **Fastest** |
| CNN | **97.80%** | Fast (185 μs/step) | Very Fast (35 μs/step) |
| MLP | 95.5% | **Very Fast (60 μs/step)** | Very Fast (42 μs/step)|
| FastText (1-gram)| 94.40% | Fast (83 μs/step) | **Very Fast (26 μs/step)** |
| FastText (2-gram)| 95.59% | Fast (196 μs/step) | **Very Fast (26 μs/step)** |
| RNN (LSTM) | **98.49%** | Very Slow (7000 μs/step) | Very Slow (1000 μs/step)|

With some hyperparameter tuning:

| Model | Accuracy | Train Speed | Classification Speed |
| -------- | ----------- | --------- | ----------------- |
| Dict | 85% | **Fastest** | **Fastest** |
| CNN | **99.40%** | Fast (200 μs/step) | Very Fast (26 μs/step) |
| MLP | 96.5% | **Very Fast (60 μs/step)** | Very Fast (42 μs/step)|
| FastText (1-gram)| 94.40% | Fast (117 μs/step) | **Very Fast (26 μs/step)** |
| FastText (2-gram)| 95.59% | Fast (196 μs/step) | **Very Fast (26 μs/step)** |
| RNN (LSTM) | **98.49%** | Very Slow (7000 μs/step) | Very Slow (1000 μs/step)|

Cofiguration, GTX 1080, 32 Gb RAM, 12x 3.6 Ghz cores, Arch Linux, up to date on 12/16/2018

## CNN Hyperparameter tuning

| Accuracy | Speed | Batch Size | Embedding Dims | Filters | Kernel | Hidden Dims | Epochs |
|--------|------------|------|-----|-----|----|-----|---|
| 99.40% | 26 μs/step |   64 |  75 | 100 |  5 | 350 | 7 |                        
| 99.36% | 40 μs/step |   64 |  50 | 250 | 10 | 150 | 5 |                      
| 99.33% | 25 μs/step |   64 |  75 |  75 |  5 | 350 | 5 |                      
| 99.31% | 59 μs/step |   64 | 100 | 350 |  5 | 300 | 3 |                      
| 99.29% | 25 μs/step |   64 |  50 | 100 |  7 | 350 | 5 |                      
| 99.27% | 62 μs/step |   32 |  75 | 350 |  5 | 250 | 3 |                      
| 99.25% | 25 μs/step |   64 |  75 | 100 |  3 | 350 | 5 |                      
| 99.25% | 25 μs/step |   64 |  50 | 100 |  7 | 250 | 3 |                      
| 99.24% | 53 μs/step |   64 |  75 | 350 | 10 | 250 | 3 |                      
| 99.23% | 56 μs/step |   64 |  75 | 350 | 10 | 200 | 3 |                      
| 99.18% | 36 μs/step |   64 |  50 | 250 |  5 | 300 | 5 |                       
| 99.12% | 52 μs/step |   64 |  75 | 350 |  5 | 250 | 3 |                       
| 99.11% | 22 μs/step |   64 |  50 |  75 |  5 | 300 | 4 |                       
| 99.11% | 26 μs/step |   64 |  50 | 100 | 10 | 250 | 3 |                      
| 99.04% | 62 μs/step |   32 |  75 | 350 |  5 | 350 | 3 |                      
| 99.00% | 24 μs/step |   64 | 100 |  50 |  5 | 350 | 3 |                      
| 99.00% | 52 μs/step |   64 |  75 | 350 |  5 | 350 | 3 |                      
| 99.00% | 40 μs/step |   64 |  75 | 250 |  5 | 350 | 3 |                      
| 98.84% | 50 μs/step |   64 |  50 | 350 | 10 | 150 | 3 |                      
| 98.86% | 40 μs/step |   64 |  75 | 250 |  5 | 250 | 3 |
| 98.79% | 26 μs/step |   64 |  50 | 100 | 10 | 150 | 3 |                      
| 98.76% | 30 μs/step |  128 |  50 | 200 |  3 | 150 | 3 |                      
| 98.66% | 31 μs/step |   64 |  50 | 150 | 10 | 150 | 3 |                      
| 98.62% | 45 μs/step |  128 | 100 | 350 |  3 | 250 | 3 |                      
| 98.17% | 19 μs/step |   64 |  75 |  50 |  3 | 350 | 6 |                      
| 98.07% | 34 μs/step |  128 |  75 | 250 |  5 | 250 | 3 |                      
| 98.06% | 45 μs/step |   64 |  75 | 350 |  3 | 250 | 3 |                      
| 97.53% | 35 μs/step |  128 |  75 | 250 |  5 | 350 | 3 |                      
| 96.10% | 32 μs/step |  128 |  75 | 250 |  3 | 350 | 3 | 