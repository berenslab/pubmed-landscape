import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split


# KNN accuracy

def knn_accuracy(Zs, colors, k=10, subset_size=500, rs=42):
    """Calculates kNN accuracy.
    Calculates the kNN accuracy, for a subset of labeled points (color different than grey), doing a train test split.
    
    Parameters
    ----------
    Zs : list of array-like
        List with the different datasets for which to calculate the kNN accuracy.
    colors : array-like
        Array with labels (colors).
    k : int, default=10
        Number of nearest neighbors to use.
    subset_size : int, default=500
        Subset size for the kNN calculation
    rs : int, default=42
        Random seed.
    
    Returns
    -------
    knn_scores : list of floats
        List with the kNN accuracy for the different datasets `Zs`.
    """
    
    knn_scores=[]
    
    for i, Xrp in enumerate(Zs):
        n = np.sum(colors!='lightgrey')
        np.random.seed(rs)
        test = np.random.choice(n, size=subset_size, replace=False)
        train = np.setdiff1d(np.arange(n), test)

        neigh = KNeighborsClassifier(n_neighbors=10, algorithm='brute', n_jobs=-1)
        neigh.fit(Xrp[colors!='lightgrey'][train], colors[colors!='lightgrey'][train])
        acc = np.mean(neigh.predict(Xrp[colors!='lightgrey'][test]) == colors[colors!='lightgrey'][test])
        knn_scores.append(acc)

    return knn_scores




def knn_accuracy_ls(selected_embeddings, true_labels, k = 10, rs=42):
    """Calculates kNN accuracy.
    In principle should do the same as the function above, but the way of selecting the train and test set is differently.
    Code from Luca.
    
    Parameters
    ----------
    selected_embeddings : list 
        List with the different datasets for which to calculate the kNN accuracy.
    true_labels : array-like
        Array with labels (colors).
    k : int, default=10
        Number of nearest neighbors to use.
    rs : int, default=42
        Random seed.
    
    Returns
    -------
    knn_accuracy : float
        kNN accuracy of the dataset.
    
    """
    
    random_state = np.random.seed(rs)
    
    X_train, X_test, y_train, y_test = train_test_split(selected_embeddings, true_labels, test_size=0.01, random_state = random_state)
    
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', n_jobs=-1)
    knn = knn.fit(X_train, y_train)
    knn_accuracy = knn.score(X_test, y_test)
    
    return knn_accuracy



# KNN recall

def knn_recall(X, Zs, k=10, subset_size=None):
    """Calculates kNN recall. 
    Calculates the kNN recall for `k`nearest neighbors.
    
    Parameters
    ----------
    X : {array-like, sparse matrix}
        High-dimensional data.
    Zs : list of array-like
        List of different low-dimensional data. When computing for only one low-dimensional dataset, still needs to be a list of array-like: [Z].
    k : int, default=10
        Number of nearest eeighbors.
    subset_size : int, optional
        Size of the subset of the data, if desired.
    
    Returns
    -------
    knn_recall : array like of shape (len(`Zs`))
        KNN recall for each of the `Zs` low-dimensional versions of the dataset.
    
    See Also
    --------
    knn_recall_affinity_matrix, knn_recall_and_ratios
    
    
    Note
    ----
    KNN recall is by definition the fraction of preserved nearest neighbors from the high-dimensional version of the dataset to the low-dimensional one.

    """
    
    if subset_size is not None:
        np.random.seed(42)
        subset = np.random.choice(X.shape[0], size=subset_size, replace=False)
        
        # In this case we will have to query k+1 points, because
        # sklearn returns the query point itself as one of the neighbors
        k_to_query = k + 1
    else:
        # In this case we can query k points
        k_to_query = k 
    
    nbrs1 = NearestNeighbors(n_neighbors=k_to_query, algorithm='brute', n_jobs=-1).fit(X)
    ind1 = nbrs1.kneighbors(X=None if subset_size is None else X[subset],
                            return_distance=False)
        
    knn_recall = np.zeros(len(Zs))
    for num, Z in enumerate(Zs):
        print('.', end='')
        nbrs2 = NearestNeighbors(n_neighbors=k_to_query, algorithm='brute', n_jobs=-1).fit(Z)
        ind2 = nbrs2.kneighbors(X=None if subset_size is None else Z[subset],
                                return_distance=False)
    
        intersections = 0.0
        for i in range(ind1.shape[0]):
            intersections += len(set(ind1[i]) & set(ind2[i]))
        
        if subset_size is None:
            knn_recall[num] = intersections / ind1.shape[0] / k
        else:
            # it substracts the ind1.shape[0] because when you take a subset of the data
            # in the NearestNeighbors.kneighbors function, it takes the query point itself as
            # one of the neighbors, so you need to substract the intersection of a point with himself
            knn_recall[num] = (intersections - ind1.shape[0]) / ind1.shape[0] / k
    
    
    return knn_recall



# Isolatedness

def knn_covid(X, mask_covid, k=10, subset_size=None):
    """Calculates isolatedness.
    Calculates the metric "isolatedness", which expresses which fraction of the kNN of a subset that also belongs to that subset. The NN algorithm is trained for all the dataset, but only papers belonging to the subset are evaluated.

    Parameters
    ----------
    X : array-like
        Dataset.
    mask_covid : list of bool
        Mask selecting the subset of papers.
    k : int, default=10
        Number of nearest neighbors.
    subset_size : int, optional 
        Subset size (different than the subset itself -- subset of the subset) for the subset of papers to be evaluated.
        
    Returns
    -------
    knn_covid : float
        Isolatedness of the subset.

        """
    
    # create necessary stuff
    X_covid = X[mask_covid]
    
    labels_covid = ['covid']*mask_covid.shape[0]
    labels_covid = np.where(mask_covid, labels_covid, 'non_covid')
    
    k_to_query = k + 1
    
    # create subset
    if subset_size is not None:
        np.random.seed(42)
        subset = np.random.choice(X_covid.shape[0], size=subset_size, replace=False)
    
    # get NN
    nbrs = NearestNeighbors(n_neighbors=k_to_query, algorithm='brute', n_jobs=-1).fit(X)
    dist , ind = nbrs.kneighbors(X=X_covid if subset_size is None else X_covid[subset],
                            return_distance=True)
    
    # here ind.shape[0] is the size of the subset
    # here ind.shape[1] is k_to_query
    
    # count covid papers in NN
    nn_covid = 0.0
    for i in range(ind.shape[0]):
        #looping over the size of the subset
        nn_covid += len(np.where(labels_covid[ind[i]] == 'covid')[0])

    # Normalize knn_covid : 
    # we substract the ind.shape[0] because when you take a subset of the data
    # in the NearestNeighbors.kneighbors function, it takes the query point itself as
    # one of the neighbors, so you need to substract the intersection of a point with himself
    knn_covid = (nn_covid - ind.shape[0]) / ind.shape[0] / k
 
    return knn_covid





# Counts and fraction of words
def counts_and_fraction_words(abstract_series, words_to_check):
    """Computes the number and the fraction of abstracts containing a word.
    
    Parameters
    ----------
    abstract_series : panda series
        Abstracts.
    words_to_check : list of str
        List with the words to check.
    
    Returns
    -------
    counts_words : array-like of shape (n_words_to_check,)
        Number of abstracts cointaining each word.
    fraction_words : array-like of shape (n_words_to_check,)
        Fraction of total abstracts containing each word.
    
    """
    
    counts_words = []
    fraction_words = []
    
    for word in words_to_check:
        sub1=' '+word
        sub2=word.capitalize()

        indexes1= abstract_series.str.find(sub1)
        indexes2= abstract_series.str.find(sub2)

        word_cnts = len(np.where((indexes1!=-1) | (indexes2!=-1))[0])
            
        counts_words.append(word_cnts)
        fraction_words.append(word_cnts/len(abstract_series))
    
    counts_words = np.array(counts_words)
    fraction_words = np.array(fraction_words)
    
    assert counts_words.shape[0] == len(words_to_check)
    
    return counts_words, fraction_words




def counts_and_fraction_words_in_red_squares(square_coordinates, tsne, abstract_series, words_to_check, verbose=True):
    """Computes the number and the fraction of abstracts containing a word, for abstracts belonging to different regions in the embedding.
    If there are no points in a region, the count and fraction will be -100. 
    
    Parameters
    ----------
    square_coordinates : array-like of shape (n_regions, 4)
        Embedding coordinates delimiting the regions in the form [top, bottom, left, right].
    tsne : array-like
        t-SNE data for the selection of the embedding regions.
    abstract_series : panda series
        Abstracts.
    words_to_check : list of str
        List with the words to check.
    verbose : bool, optional
        If true, prints the region being calculated. 
    
    Returns
    -------
    counts_words_regions : array-like of shape (n_regions, n_words_to_check)
        Number of abstracts cointaining each word.
    fraction_words_regions : array-like of shape (n_regions, n_words_to_check)
        Fraction of total abstracts containing each word.
    size_regions : array-like of shape (n_regions,)
        Number of papers per region for normalizations.
        
    See Also
    --------
    counts_and_fraction_words
    
    """
    
    counts_words_regions = []
    fraction_words_regions = []
    size_regions = []
    
    for i in range(square_coordinates.shape[0]):
        if verbose == True:
            print('Region ',i+1)
            
        coordinates = square_coordinates[i]
        
        top = coordinates[0]
        bottom = coordinates[1]
        left = coordinates[2]
        right = coordinates[3]
        
        mask_region = (tsne[:,0]<right) & (tsne[:,0]>left) & (tsne[:,1]<top) & (tsne[:,1]>bottom)
        
        abstract_region = abstract_series[mask_region]
        size_regions.append(len(abstract_region))
        
        if abstract_region.shape[0] == 0:
            counts_words_regions.append([-100]*len(words_to_check))
            fraction_words_regions.append([-100]*len(words_to_check))
    
        else:
            counts_words, fraction_words = counts_and_fraction_words(abstract_region, words_to_check)
            
            counts_words_regions.append(counts_words)
            fraction_words_regions.append(fraction_words)

    counts_words_regions = np.vstack(counts_words_regions) # potentially transform into a panda (?)
    fraction_words_regions = np.vstack(fraction_words_regions) # potentially transform into a panda (?)
    size_regions = np.array(size_regions)
    
    return counts_words_regions, fraction_words_regions, size_regions



def knn_accuracy_years(Xs, years, k=10, subset_size=500, rs=42):
    """Mean-squared error of $k$NN prediction of publication year
    This function outputs the MSE. For a metric in unit [years], compute the square root of the output (RMSE).
    
    Parameters
    ----------
    Xs : list of array-like
        List with the different datasets for which to calculate the kNN accuracy.
    years : array-like
        Array with labels (years).
    k : int, default=10
        Number of nearest neighbors to use.
    subset_size : int, default=500
        Subset size for the kNN calculation
    rs : int, default=42
        Random seed.
    
    Returns
    -------
    knn_accuracies_years : list of ints
        List with the metric values for the different datasets `Xs`.
    """
    
    knn_accuracies_years = []
    
    for X in Xs:
        n = X.shape[0]
        np.random.seed(rs)
        test = np.random.choice(n, size=subset_size, replace=False)
        train = np.setdiff1d(np.arange(n), test)

        # In this case we will have to query k+1 points, because
        # sklearn returns the query point itself as one of the neighbors
        k_to_query = k + 1

        nbrs1 = NearestNeighbors(n_neighbors=k_to_query, algorithm='brute', n_jobs=-1).fit(X[train])
        ind1 = nbrs1.kneighbors(X[test],
                                return_distance=False)

        pred_years = []
        for i in range(ind1.shape[0]):
            pred_year = np.mean(years[ind1[i,1:]])
            pred_years.append(pred_year) 

        true_years = years[ind1[:,0]]
        knn_years = mean_squared_error(true_years, pred_years)
        
    knn_accuracies_years.append(knn_years)

    return knn_accuracies_years



def chance_knn_accuracy(Zs, colors, subset_size=500, rs=42):
    """Chance kNN accuracy.
    Calculates the chance kNN accuracy, for a subset of labeled points (color different than grey), doing a train test split.
    Note that the dataset does not really matter since the dummy classifier does not look for neighbors but just randomly draws one of the labels as prediction.
    
    Parameters
    ----------
    Zs : list of array-like
        List with the different datasets for which to calculate the chance kNN accuracy.
    colors : array-like
        Array with labels (colors).
    subset_size : int, default=500
        Subset size for the kNN calculation
    rs : int, default=42
        Random seed.
    
    Returns
    -------
    knn_scores : list of ints
        List with the kNN accuracy for the different datasets `Zs`.
    """
    
    knn_scores=[]
    
    for i, Xrp in enumerate(Zs):
        n = np.sum(colors!='lightgrey')
        np.random.seed(rs)
        test = np.random.choice(n, size=subset_size, replace=False)
        train = np.setdiff1d(np.arange(n), test)

        dummy_clf = DummyClassifier(strategy="stratified", random_state=rs)
        dummy_clf.fit(Xrp[colors!='lightgrey'][train], colors[colors!='lightgrey'][train])
        acc = dummy_clf.score(Xrp[colors!='lightgrey'][test], colors[colors!='lightgrey'][test])
        knn_scores.append(acc)

    return knn_scores



def knn_accuracy_years_chance(Xs, years, k=10, subset_size=500, rs=42):
    """Mean-squared error of chance $k$NN prediction of publication year
    This function outputs the MSE. For a metric in unit [years], compute the square root of the output (RMSE).
    
    Parameters
    ----------
    Xs : list of array-like
        List with the different datasets for which to calculate the kNN accuracy. Dataset is indiferent since NN are randomly sampled. Only its size matters.
    years : array-like
        Array with labels (years).
    k : int, default=10
        Number of nearest neighbors to use.
    subset_size : int, default=500
        Subset size for the kNN calculation
    rs : int, default=42
        Random seed.
    
    Returns
    -------
    knn_accuracies_years : list of ints
        List with the metric values for the different datasets `Xs`.
    """
    
    knn_accuracies_years = []
    
    for X in Xs:
        n = X.shape[0]
        np.random.seed(rs)
        test = np.random.choice(n, size=subset_size, replace=False)
        
        true_years = years[test]
        
        pred_years = []
        for point in test:
            # like this we ensure that the point is not its own neighbor
            train = np.setdiff1d(np.arange(n), point)
            indices = train[np.random.choice(train.shape[0], size=k, replace=False)]
            
            pred_year = np.mean(years[indices])
            pred_years.append(pred_year)
            
            
        knn_years = mean_squared_error(true_years, pred_years)
        knn_accuracies_years.append(knn_years)

    return knn_accuracies_years



from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

def knn_accuracy_whitening_scores(X, y, ntest=500, rs=42):
    """Calculates kNN accuracy of raw, centered and whitened data.
    It calculates it for to distance metrics: cosine and euclidean.
    
    Parameters
    ----------
    X : list of array-like
        List with the different datasets for which to calculate the kNN accuracy.
    y : array-like
        Array with labels (colors).
    ntest : int, default=500
        Subset size for the kNN calculation
    rs : int, default=42
        Random seed.
    
    Returns
    -------
    scores : array of floates of shape (3,2)
        List with the kNN accuracy for the different distance metrics and versions of the data.
        
    """

    n=X.shape[0]
    Xcentered = X - np.mean(X,axis=0)
    Xwhitened = PCA(whiten=True).fit_transform(X)

    scores = np.zeros((3,2))
    np.random.seed(rs)
    test = np.random.choice(n, size=ntest, replace=False)
    train = np.setdiff1d(np.arange(n), test)

    for i, X_to_use in enumerate([X, Xcentered, Xwhitened]):
        for j, metric in enumerate(['euclidean', 'cosine']):

            knn = KNeighborsClassifier(
                n_neighbors=10,
                algorithm='brute',
                metric=metric
            ).fit(
                X_to_use[train],
                y[train]
            )

            scores[i,j] = knn.score(X_to_use[test], y[test])

    return scores