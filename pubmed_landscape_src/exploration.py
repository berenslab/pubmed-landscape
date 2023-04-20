import pandas as pd
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances


def print_tfidf_top_words(tfidf_features, vocabulary_swap, mask, n=5):
    """Prints top TF-IDF words.
    Prints the `n` words with the highest mean TF-IDF score in the region defined by the `mask`.
    
    Parameters
    ----------
    tfidf_features : sparse matrix
        TF-IDF features.   
    vocabulary_swap : dict
        Reverse dictionary of the TF-IDF vocabulary, {index : word}.
    mask : array of bool
        Mask of the selected region.
    n : int, default=5
        Number of words to be printed.
        
    See Also
    --------
    print_tfidf_threshold_words : Prints the most frequent TF-IDF words above a certain threshold.
    
    """
    
    tfidf_features_reduced=tfidf_features[mask,:]
  
    print('There are {} papers.'.format(tfidf_features_reduced.shape[0]))
    mean_tfidf_values = np.array(np.mean(tfidf_features_reduced, axis=0)).flatten()
    
    sorted_tfidf_indeces = np.flip(np.argsort(mean_tfidf_values))
    sorted_tfidf_values = mean_tfidf_values[sorted_tfidf_indeces]
    sorted_tfidf_columns = np.arange(tfidf_features_reduced.shape[1])[sorted_tfidf_indeces]
    words=np.vectorize(vocabulary_swap.get)(sorted_tfidf_columns[:n])
    print('TF-IDF: ', sorted_tfidf_values[:n])
    print('Words: ',words)

    
def print_tfidf_threshold_words(tfidf_features, vocabulary_swap, mask, n, several_th = False):
    """Prints the most relevant TF-IDF words.
    Prints the most frequent `n` words with TF-IDF values above a certain threshold (th=0.10), in the region defined by the `mask`. 
    
    Parameters
    ----------
    tfidf_features : sparse matrix
        TF-IDF features.
    vocabulary_swap : dict
        Reverse dictionary of the TF-IDF vocabulary, {index : word}.
    mask : array of bool
        Mask of the selected region.
    n : int, default=5
        Number of words to be printed.
    several_th : bool, default = False
        If True, the words are printed for several values of threshold (th=[0.05, 0.10, 0.15]).
    
    See Also
    --------
    print_tfidf_top_words : Prints the TF-IDF words with the highest mean TF-IDF score.
    
    Notes
    -----
    All words/tfidf-elements above a certain threshold are selected. The goal of this threshold is to eliminate frequent low-content words (like stopwords) which tent to have a relatively low tf-idf score (below 0.05). Then, from all the words with tf-idf scores above a certain threshold, the most frequent words are selected, to ensure that the words printed appear in a majority of the papers of the chosen island, and thus provide a good overview of the common topic of the island.

    If we were just to select the words with the highest tf-idf score, we would get mostly very rare words, sometimes even numeric terms (a specific cell culture, a specifig drug name/code...). These words don't provide a good overview of the topic of the abstracs. Therefore, we choose a threshold with a lower value of the tf-idf score. The most informative thresholds are around 0.05-0.15. After the threshold, by choosing the most frequent words, we ensure that the words printed occur in many of the papers of the area that we have selected. It is also important selecting a tf-idf threshold instead of a fixed value, since a word may have different tf-idf scores in different papers.
    
    """
        
    tfidf_features_reduced=tfidf_features[mask,:]
    print('There are {} papers.'.format(tfidf_features_reduced.shape[0]))
    tfidf_nonzero=sp.sparse.find(tfidf_features_reduced)
    nonzero_elem=np.array(list(zip(tfidf_nonzero[0], tfidf_nonzero[1], tfidf_nonzero[2])))
    
    if several_th == False:
        threshold = [0.1]
        
    if several_th == True:
        threshold=np.arange(0.05,0.20,0.05)
        
    for i, th in enumerate(threshold):
        mask=nonzero_elem[:,2]>=th
        greater_elem=nonzero_elem[mask]

        rows=greater_elem[:,0]
        unique_rows=np.unique(rows)
        print('In this threshold there are {} papers.'.format(len(unique_rows)))

        columns=greater_elem[:,1]
        unique_columns,count_columns=np.unique(columns,return_counts=True)


        sorted_counts_indeces=np.flip(np.argsort(count_columns))
        sorted_columns=np.take_along_axis(unique_columns,sorted_counts_indeces,0)
        sorted_counts=np.flip(np.sort(count_columns))
        print('For threshold = {:.2f}'.format(th),sorted_counts[:n])    

        if sorted_columns.size > 0:
            words=np.vectorize(vocabulary_swap.get)(sorted_columns[:n])
            print('For threshold = {:.2f}'.format(th),words)
    
    
def find_and_print_NN(serie_titles, dataset, paper_index, k=10, print_abstracts=False, serie_abstracts=None):
    """Prints the titles of the nearest neighbors.
    Finds and prints the titles of the `k`-NN of one particular paper. If `print_abstracts` = True and `serie_abstracts` is given, it also prints the abstracts. It doesn't support sparse datasets
    
    Parameters
    ----------
    serie_titles : pandas series
        Titles.
    dataset : array-like
        Data in which the neighbors are searched.
    paper_index : int
        Index of the paper to query.
    k : int, default=10. 
        Number of nearest neighbors to search.
    print_abstracts : bool, optional
        If True along with a given `serie_abstracts`, it will also print the abstracts.
    - serie_abstracts : pandas series, default=None
        Abstracts. If given along with `print_abstracts`=True, it will also print the abstracts.
    
    Returns
    -------
    indeces_k1nn : array-like of int
        Indeces of the the `k`+1 nearest neighbors (the first index corresponds to the point queried).
        
    See Also
    --------
    find_and_print_NN_sparse : analogous function for sparse datasets.
    
    """
    
    dataset_paper=dataset[paper_index,:]
    print(dataset.shape)
    print(dataset_paper.shape)
    d = cdist(dataset,dataset_paper.reshape(1,-1))
    indeces_k1nn=np.argsort(d.flatten())[:k+1]
    
    #print
    for i in range(0,len(indeces_k1nn)):
        print('Neighbor {:}:'.format(i),serie_titles.iloc[indeces_k1nn].tolist()[i])
        #print(serie_titles.iloc[indeces_k1nn].tolist()[i])
        if print_abstracts == True:
            print(serie_abstracts.iloc[indeces_k1nn].tolist()[i])
            print('----------------------------------------------------------')
    
    return indeces_k1nn


def find_and_print_NN_sparse(serie_titles, dataset, paper_index, k=10, print_abstracts=False,serie_abstracts=None):
    """Prints the titles of the nearest neighbors.
    Finds and prints the titles of the `k`-NN of one particular paper. If `print_abstracts` = True and `serie_abstracts` is given, it also prints the abstracts. It supports sparse datasets
    
    Parameters
    ----------
    serie_titles : pandas series
        Titles.
    dataset : {array-like, sparse matrix}
        Data in which the neighbors are searched.
    paper_index : int
        Index of the paper to query.
    k : int, default=10. 
        Number of nearest neighbors to search.
    print_abstracts : bool, optional
        If True along with a given `serie_abstracts`, it will also print the abstracts.
    - serie_abstracts : pandas series, default=None
        Abstracts. If given along with `print_abstracts`=True, it will also print the abstracts.
    
    Returns
    -------
    indeces_k1nn : array-like of int
        Indeces of the the `k`+1 nearest neighbors (the first index corresponds to the point queried).
        
    """
        
    dataset_paper=dataset[paper_index,:]
    print(dataset.shape)
    print(dataset_paper.shape)
    d = pairwise_distances(dataset,dataset_paper.reshape(1,-1))
    indeces_k1nn=np.argsort(d.flatten())[:k+1]
    
    #print
    for i in range(0,len(indeces_k1nn)):
        print('Neighbor {:}:'.format(i),serie_titles.iloc[indeces_k1nn].tolist()[i])
        if print_abstracts==True:
            print(serie_abstracts.iloc[indeces_k1nn].tolist()[i])
            print('----------------------------------------------------------')
    
    return indeces_k1nn


def find_mask_words(abstracts, word, verbose=True):
    """ Creates a mask for abstracts containing a certain word.
    Creates several masks of the size of `abstracts` for instances containing the words in `words`. Also it prints how many instances contain each word, in its capitalized, uncapitalized versions, and total.
    
    Parameters
    ----------
    abstracts : pandas dataframe of str
        All texts (in this case abstracts).
    words : str
        str of the word/phrase to be queried.
    verbose : bool, optional
        If True, prints the number of times the word appears in its different forms in the abstracts collection.
        
    Returns
    -------
    mask : array-like of bool
        Mask.
    
    """
    
    sub1=' '+word
    sub2=word.capitalize()

    indexes1= abstracts.str.find(sub1)
    indexes2= abstracts.str.find(sub2)

    mask = (indexes1!=-1) | (indexes2!=-1) 

    if verbose == True:
        print(f"Number of papers with uncapitalized word '{word}': ", len(np.where(indexes1!=-1)[0]))
        print(f"Number of papers with capitalized word '{word}': ", len(np.where(indexes2!=-1)[0]))
        print(f"Number of total papers with word '{word}': ", len(np.where(mask)[0]))
    
    return mask



def print_numbers_names(names_first_author, gender_first_author, names_last_author, gender_last_author):
    """Prints some statistics on available, predicted and female (first/last) author names.
    Returns the absolute numbers and percentages of available and predicted names, and female authors for both first and last authors.
    
    Parameters
    ----------
    names_first_author : ndarray
        Names of first authors.
    gender_first_author : ndarray
        Genders of first authors.
    names_last_author : ndarray
        Names of last authors.
    gender_last_author : ndarray
        Genders of last authors.
    """
    
    assert names_first_author.shape[0] == names_last_author.shape[0]
    total = names_first_author.shape[0]
    
    
    print('Number of available first author names: ', len(np.where(names_first_author != '')[0]))
    print('Number of predicted first author names: ', len(np.where(gender_first_author != 'unknown')[0]))
    print('Number of female first author names: ', len(np.where(gender_first_author == 'female')[0]))

    print('% of available first author names: ', len(np.where(names_first_author != '')[0])/total*100)
    #print('% of predicted first author names out of total: ', len(np.where(gender_first_author != 'unknown')[0])/total*100)
    print('% of predicted first author names: ', len(np.where(gender_first_author != 'unknown')[0])/len(np.where(names_first_author != '')[0])*100)
    print('% of female first author names: ', len(np.where(gender_first_author == 'female')[0])/len(np.where(gender_first_author != 'unknown')[0])*100)

    print('Number of available last author names: ', len(np.where(names_last_author != '')[0]))
    print('Number of predicted last author names: ', len(np.where(gender_last_author != 'unknown')[0]))
    print('Number of female last author names: ', len(np.where(gender_last_author == 'female')[0]))

    print('% of available last author names: ', len(np.where(names_last_author != '')[0])/total*100)
    print('% of predicted last author names: ', len(np.where(gender_last_author != 'unknown')[0])/len(np.where(names_first_author != '')[0])*100)
    print('% of female last author names: ', len(np.where(gender_last_author == 'female')[0])/len(np.where(gender_last_author != 'unknown')[0])*100)


    


def print_numbers_names_label(label, names_first_author, gender_first_author, names_last_author, gender_last_author, colors_new, colors_new_legend):
    """Prints some statistics on available, predicted and female (first/last) author names for a given label
    Returns the absolute numbers and percentages of available and predicted names, and female authors for both first and last authors, for a given label.
    Parameters
    ----------
    label : str
        Chosen label.
    names_first_author : ndarray
        Names of first authors.
    gender_first_author : ndarray
        Genders of first authors.
    names_last_author : ndarray
        Names of last authors.
    gender_last_author : ndarray
        Genders of last authors.
    colors_new : array
        Colors/labels.
    colors_new_legend : dict
        Legend mapping the colors to the labels.
    
    See Also
    --------
    print_numbers_names
    """
    
    names_first_author = names_first_author[colors_new == colors_new_legend[label]]
    gender_first_author = gender_first_author[colors_new == colors_new_legend[label]]
    names_last_author = names_last_author[colors_new == colors_new_legend[label]]
    gender_last_author = gender_last_author[colors_new == colors_new_legend[label]]
    
    print_numbers_names(names_first_author, gender_first_author, names_last_author, gender_last_author)