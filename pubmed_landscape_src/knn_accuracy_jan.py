def get_knn_score_split(Zs, colors, k=10, subset_size=500, rs=42):
    """Calculates kNN accuracy.
    Calculates the kNN accuracy, for a subset of labeled points (color different than grey), doing a train test split.
    
    Parameters
    ----------
    Zs : list of array-like
        List with the different datasets for which to calculate the kNN accuracy.
    colors : array-like
        Array with labels (colors in this case).
    k : int, default=10
        Number of nearest neighbors to use.
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
        n = Xrp.shape[0]
        np.random.seed(rs)
        test = np.random.choice(n, size=subset_size, replace=False)
        train = np.setdiff1d(np.arange(n), test)

        neigh = KNeighborsClassifier(n_neighbors=10, algorithm='brute')
        neigh.fit(Xrp[train], colors[train])
        acc = np.mean(neigh.predict(Xrp[test]) == colors[test])
        knn_scores.append(acc)

    return knn_scores