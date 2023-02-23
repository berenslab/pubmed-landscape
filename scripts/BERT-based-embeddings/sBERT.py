# imports
import pandas as pd
import random
import numpy as np
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer
import time
import torch

# specify & check gpu usage
device = "cuda" if torch.cuda.is_available() else "cpu" # put cuda:0 if else not working
print("running on device: {}".format(device))

df = pd.read_pickle('/mnt/qb/work/ludwig/lqb424/datasets/df_labeled_papers_subset')
df = df.reset_index(drop=True)
abstracts = df['AbstractText'].tolist()

# random seed
random_state = random.seed(42)

# loading sentence transformer model
embedder = SentenceTransformer('all-mpnet-base-v2')

# start timer
st = time.time()

# generate embeddings
abstracts_embeddings = embedder.encode(abstracts)

# get the exectution time
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

# compute knn accuracy
selected_embeddings = abstracts_embeddings[df.Colors.index]

# corresponding labels
true_labels = np.array(df.Colors)

# splitting arrays or matrices into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(selected_embeddings, true_labels, test_size=0.01, random_state = random_state)

# compute knnn accuracy
k = 10
knn = KNeighborsClassifier(n_neighbors=k)
knn = knn.fit(X_train, y_train)
print('Test accuracy) k={} is {}.'.format(k, round(knn.score(X_test, y_test),5)))

