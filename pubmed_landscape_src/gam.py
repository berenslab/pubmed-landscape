import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pygam import GAM, LinearGAM, LogisticGAM, PoissonGAM, InvGaussGAM, s, f, te 
from pygam import intercept
from .exploration import find_mask_words


def get_females_per_year(gender_first_author, year_first_author, years, subset_size = None, rs=42, verbose=False):
    """ Returns instances (for training a GAM) from each paper: its year and gender of author.
    It returns X and y to train a GAM to fit the fraction of female authors per year, where X is the year of the paper and y is whether for that paper the author was female (1) or male (0).
    If for a given year there are no papers from that year (e.g. subset of 'bioinformatic' papers in 1970), nothing is returned for that year.
    
    Parameters
    ----------
    gender_first_author : array-like of shape (n_papers,)
        List with the predicted genders for each paper ('male', 'female' or 'unknown').
    year_first_author : array-like of shape (n_papers,)
        List with the year for each paper.
    years : array-like
        List of years that you want to query.
    subset_size : int , optional 
        If not given, it will return all existing instances ('female' or 'male') from the years specified in `years`.
        If given, it will return a subset of instances of size `subset_size` for every year. 
        If for a year there is a number of instances smaller than the `subset_size`, it will ignore the instances from that year.
    rs : int, default=42
        Random seed for the subset, in case `subset_size` is given.
        
    Returns
    -------
    X_years : ndarray of shape (n_instances)
        Individual years.
    y_genders : ndarray of bool
        Individual genders ('female'= True, 'male'=False)
    average_fraction_female: ndarray of shape (n_years)
        Average fraction of female authors per years.
    average_number_females : ndarray of shape (n_years)
        Number of female authors per year.
    
    """
    
    y_genders = []
    X_years = []
    average_fraction_female = []
    average_number_females = []

    for year in years:
        if verbose == True:
            print('Year ', year)
            
        if subset_size is not None:
            genders_year = gender_first_author[year_first_author == year]
        
            if len(genders_year) < subset_size:
                continue

            np.random.seed(rs)
            subset = np.random.choice(genders_year.shape[0], size=subset_size, replace=False)
            genders_subset = genders_year[subset]

            total_pred_authors_subset = len(np.where(genders_subset != 'unknown')[0])

            if total_pred_authors_subset == 0:
                continue

            average_fraction_female.append(len(np.where(genders_subset == 'female')[0])/total_pred_authors_subset)
            average_number_females.append(len(np.where(genders_subset == 'female')[0]))

            genders_subset_predicted = genders_subset[genders_subset != 'unknown']

            X_years.append([year]*len(genders_subset_predicted))
            y_genders.append(np.where(genders_subset_predicted == 'female', 1, 0))

        else:
            genders_year = gender_first_author[year_first_author == year]

            total_pred_authors_year = len(np.where(genders_year != 'unknown')[0])

            if total_pred_authors_year == 0:
                continue

            average_fraction_female.append(len(np.where(genders_year == 'female')[0])/total_pred_authors_year) 
            average_number_females.append(len(np.where(genders_year == 'female')[0]))

            genders_year_predicted = genders_year[genders_year != 'unknown']

            X_years.append([year]*len(genders_year_predicted))
            y_genders.append(np.where(genders_year_predicted == 'female', 1, 0))


    y_genders = np.hstack(y_genders)
    X_years = np.hstack(X_years)

    average_fraction_female = np.array(average_fraction_female)
    average_number_females = np.array(average_number_females)

    return X_years, y_genders, average_fraction_female, average_number_females


def get_knn_overlap_gam(X, labels, date_year, years, class1, other_classes, k=10, subset_size=None, rs=42, verbose=False):
    """Returns instances (for training a GAM) for each paper: its year and its kNN overlapp to one class.
    The X are the individual years and the y are the kNN overlap of each paper with one other class. It returns one y (columns of the array) for each class given in `other_classes`.
    For the years where there are no `class1` points, the corresponding X and y values will be -1. 
    If subset_size= not None, and n_class1_points < `subset_size`, then it will take all available points.  
    Take this into account to filter this values for plotting after that. 
    
    Parameters
    ----------
    X : array-like of shape (n_papers, n_dims)
        Dataset where to look for the neighbors.
    labels : array-like of shape (n_papers,)
        Labels of papers.
    date_year : array-like of shape (n_papers,)
        Years of papers.
    years : array-like
        Years in which to query.
    class1 : str
        Main class (label).
    other_classes : list of str
        Other labels for which to calculate the kNN overlap.
    k : int, default=10
        Number of nearest neighbors to query.
    subset_size : int, default=None
        If true, only a subset of the points of size `subset_size` will be taken for every year.
    rs : int, default=42
        Random seed for the selection of the subset.
    verbose : bool, default=False
        If True, print year being calculated.
    
    Returns
    -------
    individual_years : ndarray of shape (n_points,)
        Individual years of the instances. n_points is either the subset of points with label == `class1` or `subset_size`*len(`years`).
    individual_knn_overlapp : ndarray of shape (n_points, n_other_classes)
        Individual kNN overlap of the instances, for every class (columns).
    average_knn_overlap: (n_years, n_other_classes)
        Average kNN overlap, for every class (columns). 
    """
    
    # select only labeled papers
    X_labeled = X[labels != 'unlabeled']
    labels_labeled = labels[labels != 'unlabeled']
    date_year_labeled = date_year[labels != 'unlabeled']

    # specify k_to_query and initialize variables
    if subset_size is not None:
        # In this case we will have to query k+1 points, because
        # sklearn returns the query point itself as one of the neighbors
        k_to_query = k + 1
        
        # initialize variables
        individual_years = np.ones((len(years))*subset_size)*-1
        individual_knn_overlapp = np.ones(shape=((len(years))*subset_size,len(other_classes)))*-1
        average_knn_overlap = np.ones(shape=((len(years)),len(other_classes)))*-1
        
    else:
        # In this case we can query k points
        k_to_query = k
        
        # initialize variables
        n_class1_papers = X_labeled[labels_labeled == class1].shape[0]
        individual_years = np.ones(n_class1_papers)*-1
        individual_knn_overlapp = np.ones(shape=(n_class1_papers,len(other_classes)))*-1
        average_knn_overlap = np.zeros(shape=((len(years)),len(other_classes)))


    #print(individual_years.shape)
    #print(individual_knn_overlapp.shape)
    #print(average_knn_overlap.shape)
    
    # loop over decades
    n=0
    for i in range(len(years)):
        if verbose == True:
            print('year', years[i])

        # select class1 subset from a particular decade
        X_class1 = X_labeled[(labels_labeled == class1) & (date_year_labeled == years[i])]
        
        # if subset of nsc papers from that year is == 0
        if X_class1.shape[0] == 0:
            average_knn_overlap[i,:] = -1
            continue

        # create subset
        if subset_size is not None:
            # if subset of nsc papers from that year is < subset_size
            if X_class1.shape[0] < subset_size:
            #    continue
                np.random.seed(rs)
                subset = np.random.choice(X_class1.shape[0], size=X_class1.shape[0], replace=False)
            else:
                np.random.seed(rs)
                subset = np.random.choice(X_class1.shape[0], size=subset_size, replace=False)

        # get NN
        nbrs = NearestNeighbors(n_neighbors=k_to_query, n_jobs=-1).fit(X_labeled)
        dist , ind = nbrs.kneighbors(X=X_class1 if subset_size is None else X_class1[subset],
                                return_distance=True)
        # here ind.shape[0] is the size of the subset
        # here ind.shape[1] is k_to_query


        # count papers in NN
        for j in range(ind.shape[0]):
            individual_years[n+j] = years[i]


            for l, class_l in enumerate(other_classes):
                individual_knn_overlapp[n+j, l] = len(np.where(labels_labeled[ind[j]] == class_l)[0])/k
                # it doesn't matter that I divide by k=10 even though ind has 11 neighbors since the first neighbor (the point itself)
                # is always going to be from class_1 and therefore != class_l

                average_knn_overlap[i,l] += len(np.where(labels_labeled[ind[j]] == class_l)[0])
                
        n += ind.shape[0]

        # normalize average knn overlapp
        average_knn_overlap[i,:] = average_knn_overlap[i,:]/ ind.shape[0] / k

    return individual_years, individual_knn_overlapp, average_knn_overlap



def get_ml_per_year(years, abstracts, date_year, verbose=False):
    """Returns instances (for training a GAM) from each paper: its year and whether it has 'machine learning' in its abstract or not.
    It returns X and y to train a GAM to fit the fraction of ML papers per year, where X is the year of the paper and y is whether it has 'machine learning' in its abstract or not.
    If for a given year there are no papers from that year (e.g. subset of 'bioinformatic' papers in 1970), nothing is returned for that year.
    
    Parameters
    ----------
    years : array-like
        List of years that you want to query.
    abstracts : array-like of str
        Corpus of abstracts.
    date_year: array-like
        Years of the papers.
    verbose : bool, default=False
        If True, it prints the year being queried.
        
    Returns
    -------
    X_years : ndarray of shape (n_instances)
        Individual years.
    y_ml : ndarray of bool
        Contains ML or not.
    fraction_ml: ndarray of shape (n_years)
        Average fraction of papers with ML in their abstract per years.
    number_papers : ndarray of shape (n_years)
        Number of papers with ML in their abstract per year.
        
    """
    
    X_years = []
    y_ml = []
    fraction_ml = []
    number_papers=[]

    for year in years:
        if verbose == True:
            print('Year', year)

        abstracts_year = abstracts[date_year == year]
        
        if abstracts_year.shape[0]== 0:
            continue

        mask_ml = find_mask_words(abstracts_year, 'machine learning', verbose=False)

        X_years.append([year]*len(mask_ml))
        y_ml.append(mask_ml)
        fraction_ml.append(np.mean(mask_ml))
        number_papers.append(len(abstracts_year))
    
    X_years = np.hstack(X_years)
    y_ml = np.hstack(y_ml)
    fraction = np.hstack(fraction_ml)
    number_papers = np.hstack(number_papers)
    
    return X_years, y_ml, fraction_ml, number_papers


def train_logistic_gam(X_train, y_train, verbose=False):
    """Trains a Logistic GAM.
    It has the parameter values already chosen for our experiment.
    
    Parameters
    ----------
    X_train
    y_train
    verbose : bool, default=False
        If True, prints the gam.summary()
    
    Return
    ------
    gam  
        Trained gam model
    """

    n_features = 1 # number of features used in the model

    lams = np.logspace(-5,5,20) * n_features 
    splines = 12 # number of splines we will use

    gam = LogisticGAM(s(0, n_splines=splines))

    gam.gridsearch(X_train, y_train, lam=lams)
    
    if verbose == True:
        gam.summary()
    
    return gam


def train_linear_gam(X_train, y_train, verbose=False):
    """Trains a Linear GAM.
    It has the parameter values already chosen for our experiment.
    
    Parameters
    ----------
    X_train
    y_train
    verbose : bool, default=False
        If True, prints the gam.summary()
    
    Return
    ------
    gam  
        Trained gam model
    """
    
    n_features = 1 # number of features used in the model

    lams = np.logspace(-5,5,20) * n_features 
    splines = 6 # number of splines we will use

    gam = LinearGAM(s(0, n_splines=splines))

    gam.gridsearch(X_train, y_train, lam=lams)
    
    if verbose == True:
        gam.summary()
    
    return gam


def get_plot_gam(gam, gam_type={'linear', 'logistic'}):
    """Gets the necessary things from a trained GAM model to produce the plot.
    
    Parameters
    ----------
    gam  
        Trained GAM model.
    gam_type : str, {'linear', 'logistic'}
        Type of the trained GAM.

    Return
    ------
    XX
        X generated coordinates.
    pdep
        Predicted GAM values for the `XX` coordinates.
    confi : ndarray of shape (n_XX, 2)
        Confidence intervals.
    intercept
        Intercept value.
    """

    XX = gam.generate_X_grid(term=0)
    #pdep, confi = gam.partial_dependence(term=0, X=XX, width=width)
    
    if gam_type == 'logistic':
        pdep = gam.predict_proba(XX)
        
    if gam_type == 'linear':
        pdep = gam.predict(XX)
    
    confi = gam.confidence_intervals(XX)
    
    intercept = gam.coef_[-1]
    
    return XX, pdep, confi, intercept