# imports
import pandas as pd
import random
import numpy as np
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoConfig

## computing embeddings

# specify & check gpu usage
device = "cuda" if torch.cuda.is_available() else "cpu" 
print("running on device: {}".format(device))

# data
df = pd.read_pickle('/mnt/qb/work/ludwig/lqb424/datasets/df_labeled_papers_subset')
df = df.reset_index(drop=True) 
abstracts = df['AbstractText'].tolist()

# random seed
random_state = random.seed(42)

# specifying model
checkpoint = "dmis-lab/biobert-v1.1"
#"microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
#"allenai/scibert_scivocab_uncased" 
#"bert-base-uncased"
#"allenai/specter"
#"princeton-nlp/unsup-simcse-bert-base-uncased"

print("Model: {}".format(checkpoint))  

# simple wrappers 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

# set device 
model = model.to(device)

# function to compute embeddings
def embedding(abstracts: List[str]) -> torch.Tensor:
    """
    param abstracts: a batch of abstracts
    returns: tuple of hidden representations of each sample's SEP token, CLS token and average representation.
    Dimensions:  (batch_size * hidden_size)
    """
    # tokenize
    tokens = tokenizer(abstracts, padding=True, max_length=512, truncation=True, return_tensors='pt').to(device)
    # run model, get matrix of embeddings as output
    outputs = model(**tokens)[0].cpu().detach()
    vec_av = torch.mean(outputs, [0, 1]).numpy()
    vec_sep = outputs[:,-1,:].numpy()
    vec_cls = outputs[:,0, :].numpy() 

    return vec_av, vec_sep , vec_cls

embedding_av = np.empty([len(abstracts), 768])
embedding_sep = np.empty([len(abstracts), 768]) 
embedding_cls = np.empty([len(abstracts), 768])

# start timer
st = time.time()

n = 100000 
# calling function 'embedding' for batches of abstracts
for i in np.arange(len(abstracts)):
    av, sep, cls = embedding(abstracts[i])
    embedding_av[i] = av
    embedding_sep[i] = sep
    embedding_cls[i] = cls

    if i % n == 0:
        print('iteration: {}'.format(i))
	
# get the exectution time
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

# corresponding labels
true_labels = np.array(df.Colors)
selected_embeddings = embedding_av[df.Colors.index]

# corresponding labels
true_labels = np.array(df.Colors)

# splitting arrays or matrices into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(selected_embeddings, true_labels, test_size=0.01, random_state = random_state)

# compute knnn accuracy
k = 10
knn = KNeighborsClassifier(n_neighbors=k) 
knn = knn.fit(X_train, y_train)
print('Test accuracy for (average strategy) k={} is {}.'.format(k, round(knn.score(X_test, y_test),5)))

selected_embeddings = embedding_sep[df.Colors.index]
X_train, X_test, y_train, y_test = train_test_split(selected_embeddings, true_labels, test_size=0.01, random_state = random_state)
knn = KNeighborsClassifier(n_neighbors=k) 
knn = knn.fit(X_train, y_train)
print('Test accuracy for (sep token) k={} is {}.'.format(k, round(knn.score(X_test, y_test),5)))

selected_embeddings = embedding_cls[df.Colors.index]
X_train, X_test, y_train, y_test = train_test_split(selected_embeddings, true_labels, test_size=0.01, random_state = random_state)
knn = KNeighborsClassifier(n_neighbors=k) 
knn = knn.fit(X_train, y_train)
print('Test accuracy for (cls token) k={} is {}.'.format(k, round(knn.score(X_test, y_test),5)))



