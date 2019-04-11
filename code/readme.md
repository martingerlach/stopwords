## Stopword filtering

This repository contains the code for the computational analysis described in:

Martin Gerlach, Hanyu Shi, Luis A. N. Amaral: "A universal information-theoretic approach to the identification of stopwords".


## Introduction

A common pre-processing step in applications of natural language processing (such as topic modeling) involves filtering of 'uninformative' words (hereby denoted as stopwords) -- not only to decrease the amount of data but also with the hope of increasing the signal-to-noise ratio. There exist many different heuristics to the identification of stopwords such as manual stopword list, frequency-based approaches (i.e. removing words with high or low frequency), tfdidf, etc.

In this paper we propose an information-theoretic approach to the identification of stopwords. Starting from the bag-of-words model, i.e. we represent the corpus as counts of the number of times each word occurs in each document n(w,d), our approach consists of 2 main steps:
- Calculation of the conditional entropy of each word across all documents: H(w)
- Calculation of the expected entropy H'(w) from a random null model (words distributed randomly across documents) and calculating the difference: I(w) == H'(w) - H(w). Our premise is that words with low values of I(w) are less informative and should be removed.

We compare our approach with commonly used heuristics for the removal of stopwords:
- tfdidf
- manual
- words with high frequency (TOP)
- words with low frequency (BOTTOM)

We evaluate the efficacy of different stopword lists in the task of topic modeling.
Removing an increasing fraction of words from the corpus we track 2 metrics:
- accuracy: how much the inferred topics of the documents correspond to metadata labels (e.g. categories)
- reproducibily: how stable are the inferred topics across different 'runs' of the topic model algorithm. Solutions can differ due to stochasticity (e.g. Gibbs sampling) or different initial conditions leading to different local maxima in the likelihood landscape (e.g. Variational Bayes)

## How to run this code

The script `code/run.sh` will reproduce Figures 1 and 2 from the manuscript for the English corpus (20Newsgroup dataset):
- Characterization of the conditional entropies H(w) and H'(w); Fig.1a
- Characterization of the information-theoretic measure I(w); Fig. 2a,b
- Comparison of the derived stopword list with other traditional approaches in form of rank-correlation, jaccard index, and distribution over frequency spectrum.

This script will not reproduce the results in Figures 3 and 4 (and Supplementary Figures 7-12) due to the computational effort involved. In total, we ran 11,880 'jobs' (totalling severla thousand CPU hours) on Northwestern's [Quest cluster](https://www.it.northwestern.edu/research/user-services/quest/index.html) -- 11 different values for the fraction of stopwords removed, 10 different realizations for estimation of errorbars, 6 different stopword lists, 6 different corpora, 3 different topic model algorithms. The code used to run the bulk-analysis on the cluster will be made available in a github-repository.


Therefore, we provide two interactive notebooks which contain the basic steps of our analysis.

First, `code/notebook_example_filtering.ipynb` contains:
- functions to calculate the information-theoretic measure as well as standard statistics such as tfdidf
- construction of different stopword lists (Infor, tfidf, etc.); this allows to explore the effect of different choices of thresholds in the derivation of the stopword lists.

Second, `code/notebook_example_evaluation.ipynb` contains the basic functionality to:
- construct a particular stopword list
- filter the corpus: remove all words contained in the stopword lists
- run a topic model
- calculate evaluation statistics:
  - accuracy (overlap of topics to the document's category labels)
  - reproducibility (overlap of topics across two different runs of the topic model algorithm; this has been called 'token clustering')
  - run time, memory usage
  - topic coherence

## Output
- 20NewsGroup_stopword-statistics.csv; contains all statistics to generative different stopwords
- figures from paper:
  - figure-01.png
  - figure-02a.png
  - figure-02b.png
  - figure-02c.png
  - figure-02d.png

## Data
Data contains the 20NewsGroup dataset, which is a collection of news articles from 20 different news categories; the corpus contains D=18,803 documents and  N=3,831,559 tokens. This dataset contains only words with at least 3 characters.

In addition, it contains a manual list of stopwords (for Enlish) obtained from the package `mallet`.

## Environment

Packages needed:
+ python (3.6.3)
+ gensim (0.13.1) for ldavb topic model
+ pandas (0.24.2)
+ jupyter (1.0.0)
+ memory_profiler (0.54.0)
