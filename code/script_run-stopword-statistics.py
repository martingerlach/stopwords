import os,sys
import numpy as np
import pandas as pd

## custom packages
src_dir = os.path.join('src')
sys.path.append(src_dir)

from filter_words import run_stopword_statistics


## parameters
corpus_name = '20NewsGroup'
N_s = 10 ## number of realizations for the random null model
path_stopword_list =  os.path.join(os.pardir,'data','stopword_list_en')
## path to a manual stopword list (this one is from mallet)


filename = os.path.join(os.pardir,'data','%s_corpus.csv'%(corpus_name))
df = pd.read_csv(filename,index_col=0)
list_texts = [  [h.strip() for h in doc.split()  ] for doc in df['text']    ]

## get the statistics
df = run_stopword_statistics(list_texts,N_s=N_s,path_stopword_list=path_stopword_list)

## save the statistics
filename_save = os.path.join(os.pardir,'results','%s_stopword-statistics.csv'%(corpus_name))
df.to_csv(filename_save)