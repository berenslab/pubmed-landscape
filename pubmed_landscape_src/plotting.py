import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.stats import gaussian_kde
from pubmed_landscape_src.exploration import find_mask_words

def plot_square(coordinates, lw=1, c='r'):
    """Add squares to a plot defining regions.
    The regions are defined by the `coordinates` ([top, bottom, left, right]). 
    
    Parameters
    ----------
    coordinates : array-like of shape (n_regions, 4)
        Coordinates of the region limits given in the order [top, bottom, left, right].
    lw : float, default=1
        Linewidth.
    c : str, default= 'r'
        Color.
        
    """
    
    top = coordinates[0]
    bottom = coordinates[1]
    left = coordinates[2]
    right = coordinates[3]
    
    assert bottom < top, "Bottom and top coordinates given in the wrong order."
    assert left < right, "Left and right coordinates given in the wrong order."
    
    #horizontal lines
    plt.plot([left, right], [top, top], linewidth=lw, c=c)
    plt.plot([left, right], [bottom, bottom], linewidth=lw, c=c)
    
    #vertical lines
    plt.plot([left, left], [bottom, top], linewidth=lw, c=c)
    plt.plot([right, right], [bottom, top], linewidth=lw, c=c)


    
    
def plot_region_names(coordinates, ax, labels, fontsize=5):
    """Add region names to a plot defining regions.
    The regions are defined by the `coordinates` ([top, bottom, left, right]).

    Parameters
    ----------
    coordinates : array-like of shape (n_regions, 4)
        Coordinates of the region limits given in the order [top, bottom, left, right].
    ax :  Axes
    labels : list of str
        Labels for each region.
    fontsize : float, default=5
    """
    
    top = coordinates[:,0]
    bottom = coordinates[:,1]
    left = coordinates[:,2]
    right = coordinates[:,3]
    
    labels_splited = [list(np.flip(x.split())) for x in labels] 
    
    scale_factor_labels=2.5
    
    for i, label in enumerate(labels_splited):
        for j, elem in enumerate(label):
            if i+1 in [7,8,9]:
                if i+1 ==9:
                    ax.text(left[i]-3, (top[i]+bottom[i])/2 + fontsize*j*scale_factor_labels + 5, elem, va='top' , ha='right', fontsize=fontsize, c= 'k')
        
                else:
                    ax.text(left[i]-3, (top[i]+bottom[i])/2 + fontsize*j*scale_factor_labels, elem, va='top' , ha='right', fontsize=fontsize, c= 'k')
                    
            elif i+1 in [10,6]:
                ax.text(right[i], top[i] + fontsize*j*scale_factor_labels, elem, va='bottom' , ha='left', fontsize=fontsize, c= 'k')
                
            elif i+1 == 11:
                ax.text(right[i], top[i] + fontsize*j*scale_factor_labels, elem, va='bottom' , ha='left', fontsize=fontsize, c= 'k')
                
            elif i+1 == 12:
                ax.text(right[i]-35, bottom[i] - 12 + fontsize*j*scale_factor_labels, elem, va='top' , ha='left', fontsize=fontsize, c= 'k')
            
            elif i+1==4:
                ax.text(right[i]+3, (top[i]+bottom[i])/2 + fontsize*j*scale_factor_labels, elem, va='top' , ha='left', fontsize=fontsize, c= 'k')
            elif i+1==5:
                ax.text(right[i], top[i] + fontsize*j*scale_factor_labels + 1, elem, va='bottom' , ha='right', fontsize=fontsize, c= 'k')
                
            else:
                ax.text((left[i]+right[i])/2, top[i] + fontsize*j*scale_factor_labels + 1, elem, va='bottom' , ha='center', fontsize=fontsize, c= 'k')
                
                
                
def plot_region_numbers(coordinates, ax, labels, region_labels_numbers, fontsize=5):
    """Add region numbers to a plot defining regions.
    The regions are defined by the `coordinates` ([top, bottom, left, right]).

    Parameters
    ----------
    coordinates : array-like of shape (n_regions, 4)
        Coordinates of the region limits given in the order [top, bottom, left, right].
    ax :  Axes
    labels : list of str
        Labels for each region.
    region_labels_numbers : dict
        Dictionary mapping region labels to numbers.
    fontsize : float, default=5
    """

    for i, label in enumerate(labels):
        plt.text(coordinates[i,3], coordinates[i,1], region_labels_numbers[label], va='top' , ha='left', fontsize=6, c= 'r')
        
        
        
    
def automatic_coloring(journals, words_may, words_min, list_colors):
    """ Creates coloring based on words appearing in a list of documents.
    It creates an array with colors, assigning a color to each paper depending on whether it contains a word in its journal title from the lists of `words_may` and `words_min` or not. The colors that will be assigned are given in `list_colors`.
    
    IMPORTANT REMARK: if the journal name contains two words belonging to the word list, the color of the word
    located the latest in the list will be assigned to it (first, the first word's color is assigned and then 
    the second overwrites the first).
    
    Parameters
    ----------
    journals : dataframe of str
        Dataframe with the journal names of the papers, or any other corpus where to look for the words.
    words_may : list of str
        List of the words to look for, starting with capital letter.
    words_min : list of str
        List of the words to look for, strating with small letters.
    list_colors : list of str
        List of all the unique colors to assign.
    
    Returns
    -------
    word_colors : dict
        Legend of word-colors (which color has each word)
    journal_colors : array
        Colors for each paper.
        
    See Also
    --------
    improved_coloring 
    
    """
    
    
    N=len(words_may)
    
    dict_colors={}
    word_colors={}
    for i in range(N):
        # I create a dictionary with the legend word-color for informative purpose
        word_colors[words_may[i]]=list_colors[i]
        
        #sub1 is a string with the word in small letters
        sub1=words_min[i]
        #sub2 is a string with the word starting with capital letter
        sub2=words_may[i]
        
        indexes1= journals.str.find(sub1) 
        indexes2= journals.str.find(sub2)

        #information
        #non_1_1 are the indexes of the journal names containing sub1 (the word with small letters)
        non_1_1=indexes1[indexes1!=-1]
        #non_1_2 are the indexes of the journal names containing sub2 (the word starting with capital letter)
        non_1_2=indexes2[indexes2!=-1]
        
        #containing_journals are the journals (the whole name) containing either the word in small letter or starting 
        #with capital letter
        containing_journals=journals[(indexes1!=-1) | (indexes2!=-1)]
        containing_journals=containing_journals.to_numpy()
        
        #unique_containing_j are the unique journal names from containing_journals
        unique_containing_j=np.unique(containing_journals)
        
        #here we assign one color (the same to all) to each unique journal name containing the desired word
        for elem in unique_containing_j:
            dict_colors[elem]=list_colors[i]
    
    #create colors
    journal_colors=np.vectorize(dict_colors.get)(journals)
    
    #add grey to the rest of papers
    journal_colors=np.where(journal_colors==None,'lightgrey', journal_colors)
    journal_colors=np.where(journal_colors=='None','lightgrey', journal_colors)
    
    return word_colors, journal_colors
    

    
def improved_coloring(journals, dict_words_colors):
    """ Creates coloring based on words appearing in a list of documents.
    It creates an array with colors, assigning a color to each paper depending on whether it contains a word in its journal title from the keys in ` dict_words_colors`. 
    
    IMPORTANT REMARK: if the journal name contains two words belonging to the word list, the color of the word
    located the latest in the list will be assigned to it (first, the first word's color is assigned and then 
    the second overwrites the first).
    
    Parameters
    ----------
    journals : dataframe of str
        Dataframe with the journal names of the papers, or any other corpus where to look for the words.
    dict_words_colors : dict
        Dictionary matching words to colors (legend). The keys are the words and the values are the colors.
    
    
    Returns
    -------
    labels_with_unlabeled : list of str fo len (n_journals)
        List or labels (words) for all instances including label 'unlabeled'.
    colors : array
        Colors for each paper.
            
    See Also
    --------
    automatic_coloring
    
    """
    
    
    words=dict_words_colors.keys()
    labels=np.empty(len(journals))
    
    for i, wrd in enumerate(words):
        
        word_may = wrd.capitalize()
        word_min = ' '+wrd
        
        indexes1 = journals.str.find(word_may) 
        indexes2 = journals.str.find(word_min)
        
        labels = np.where((indexes1!=-1) | (indexes2!=-1), wrd, labels)
    
    #create colors
    colors=np.vectorize(dict_words_colors.get)(labels)
    
    #add grey to the rest of papers
    colors=np.where(colors==None,'lightgrey', colors)
    colors=np.where(colors=='None','lightgrey', colors)
    
    #change 0 for 'unlabeled'
    labels_with_unlabeled=np.where(colors=='lightgrey','unlabeled', labels)
    
    
    return labels_with_unlabeled, colors




def years_coloring(dates, years, dic):
    """Creates colors based on years of publication.
    
    Parameters
    ----------
    dates : pandas DataFrame
        The dataframe column with the date of the paper.
    years: array-like
        A list of all unique years as strings.
    dic: dict
        A dictionary where you have for each year a value in between 0 and 1 for the colormap.
    
    Returns
    -------
    date_colors : array-like
        Array of colors for each paper.
    date_year : array like
        The year contained in the str date of every paper.
    
    """
    
    N=len(years)
    
    dict_colors={}

    date_year=np.zeros(len(dates))
    
    for i in range(N):
        sub1=years[i]

        indexes1= dates.str.find(sub1) 

        non_1_1=indexes1[indexes1!=-1]
        
        date_year[indexes1!=-1]=int(years[i])
        
    #create colors
    date_colors=np.vectorize(dic.get)(date_year)
    
    return date_colors, date_year




def find_cluster_center(tsne, colors, legend, subset = True, subset_size = 500000, rs = 42):
    """Find cluster centers.
    Finds coordinates of the highest density point of points from each label, using gaussian_kde.
    
    Parameters
    ----------
    tsne: array-like of shape (n_points,2)
        t-SNE coordinates.
    colors : array-like of shape (n_points,)
        Color values for the colormap.
    legend : dict
        Legend label-color.
    subset : bool, default= True
         If True, a subset of the dataset is used for the cluster center calculations.
    subset_size : int, default=500000
        Size of the subset of the dataset used for the cluster center calculations.
    rs : int, default= 42
         Random seed.
         
    Returns
    -------
    center_cluster_coordinates_df : dataframe of shape (n_clusters, 2)
        Cluster center coordinates stored in two columns: "x" and "y".
    

    """
    
    words = list(legend.keys())
    unique_colors = np.array(list(legend.values()))
    
    if subset == True:
        np.random.seed(rs)
        assert tsne.shape[0] >= subset_size, "Subset size is smaller than dataset"
        index_subset=np.random.randint(0,tsne.shape[0],subset_size)
        tsne_subset=tsne[index_subset,:]
        colors_subset=colors[index_subset]
        
    else:
        tsne_subset=tsne
        colors_subset=colors
    
    # calculate cluster centers
    center_cluster_coordinates = []
    for i in range(len(words)):
        cluster=tsne_subset[colors_subset==unique_colors[i]]
        assert cluster.shape[0] > 0
        #center with kernel density
        kde = gaussian_kde(cluster.T)
        center_cluster_coordinates.append(cluster[kde(cluster.T).argmax()])
    
    center_cluster_coordinates = np.vstack(center_cluster_coordinates)
    
    center_cluster_coordinates_df = pd.DataFrame(center_cluster_coordinates, index = words, columns = ['x', 'y'])
    
    return center_cluster_coordinates_df



def plot_label_tags(tsne, colors, legend, x_lim, y_lim, ax=None, middle_value = 0, subset = True, subset_size = 500000, rs = 42, fontsize=7, capitalize=True):
    """Plots label tags and a line pointing to the embedding.
    The line from a label tag points to the location with higher points density of that specific label.
    
    
    Parameters
    ----------
    tsne: array-like of shape (n_points,2)
        t-SNE coordinates.
    colors : array-like of shape (n_points,)
        Color values for the colormap.
    legend : dict
        Legend label-color.
    x_lim : tuple (left, right)
        Limits of the x-axis.
    y_lim : tuple (bottom, top)
        Limits of the y-axis.
    ax : axes, optional
        Axes where to draw the figure. If ax=None, axes will be created. 
    middle_value : float, default=0
         The x value to decide which labels go to the left and which go to the right.
    subset : bool, default= True
         If True, a subset of the dataset is used for the cluster center calculations.
    subset_size : int, default=500000
        Size of the subset of the dataset used for the cluster center calculations.
    rs : int, default= 42
         Random seed.
    fontsize: int, default=7
         Fontsize for the labels.
    capitalize : bool, default = True
        If True, it will capitalize the labels.
    
    See Also
    --------
    find_cluster_center
    
    """
    
    assert x_lim[0] < x_lim[1], "xlim values are in the wrong order"
    assert y_lim[0] < y_lim[1], "ylim values are in the wrong order"
    
    if ax is None:
        fig, ax = plt.subplots()

    # calculate cluster centers
    center_cluster_coordinates = find_cluster_center(tsne, colors, legend, subset, subset_size, rs)
    
    # sort by x
    center_cluster_coordinates_left = center_cluster_coordinates[center_cluster_coordinates.x < middle_value].copy()
    center_cluster_coordinates_right = center_cluster_coordinates[center_cluster_coordinates.x >= middle_value].copy()

    # sort by y
    center_cluster_coordinates_left.sort_values(by = 'y', inplace=True, ascending = False)
    center_cluster_coordinates_right.sort_values(by = 'y', inplace=True, ascending = False)
    
    sorted_labels_left = center_cluster_coordinates_left.index.tolist()
    sorted_labels_right = center_cluster_coordinates_right.index.tolist()

    sorted_colors_left = np.vectorize(legend.get)(sorted_labels_left)
    sorted_colors_right = np.vectorize(legend.get)(sorted_labels_right)
    
    if capitalize == True:
        sorted_labels_left = [elem.capitalize() for elem in sorted_labels_left]
        sorted_labels_right = [elem.capitalize() for elem in sorted_labels_right]


    # PLOT
    # left
    n_left=len(sorted_labels_left)
    x=x_lim[0]*np.ones(n_left)
    y=np.linspace(y_lim[1], y_lim[0], n_left)

    for i, colr in enumerate(sorted_colors_left):
        if any( [colr=='black', colr=='#0000A6', colr=='#5A0007', colr=='#4A3B53', colr=='#1B4400',
                 colr=='#004D43', colr=='#013349', colr=='#000035', colr=='#300018', colr=='#001E09',
                 colr=='#372101', colr=='#6508ba'] ):            
            # white colored letters
            ax.text(x[i], y[i], sorted_labels_left[i], c='lightgrey', fontsize=fontsize, ha='right', bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
            ax.plot([x[i],center_cluster_coordinates_left.x[i]],[y[i],center_cluster_coordinates_left.y[i]], c=colr, linewidth=0.4, clip_on=False)
        else:
            # black colored letters
            ax.text(x[i], y[i], sorted_labels_left[i], c='black', fontsize=fontsize, ha='right', bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
            ax.plot([x[i],center_cluster_coordinates_left.x[i]],[y[i],center_cluster_coordinates_left.y[i]], c=colr, linewidth=0.4, clip_on=False)

    # right
    n_right=len(sorted_labels_right)
    x=x_lim[1]*np.ones(n_right)
    y=np.linspace(y_lim[1], y_lim[0], n_right)

    for i, colr in enumerate(sorted_colors_right):
        # color blanco
        if any( [colr=='black', colr=='#0000A6', colr=='#5A0007', colr=='#4A3B53', colr=='#1B4400',
                 colr=='#004D43', colr=='#013349', colr=='#000035', colr=='#300018', colr=='#001E09',
                 colr=='#372101', colr=='#6508ba'] ):
            ax.text(x[i], y[i], sorted_labels_right[i], c='lightgrey', fontsize=fontsize, ha='left', bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
            ax.plot([x[i],center_cluster_coordinates_right.x[i]],[y[i],center_cluster_coordinates_right.y[i]], c=colr, linewidth=0.4, clip_on=False)
        else:
            ax.text(x[i], y[i], sorted_labels_right[i], c='black', fontsize=fontsize, ha='left', bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
            ax.plot([x[i],center_cluster_coordinates_right.x[i]],[y[i],center_cluster_coordinates_right.y[i]], c=colr, linewidth=0.4, clip_on=False)
            
            
            
            
def plot_nsc_label_tags(tsne, colors, legend, x_lim, y_lim, ax=None, middle_value = 0, subset = True, subset_size = 500000, rs = 42, fontsize=7, capitalize=True):
    """Plots label tags and a line pointing to the embedding.
    The line from a label tag points to the location with higher points density of that specific label.
    
    
    Parameters
    ----------
    tsne: array-like of shape (n_points,2)
        t-SNE coordinates.
    colors : array-like of shape (n_points,)
        Color values for the colormap.
    legend : dict
        Legend label-color.
    x_lim : tuple (left, right)
        Limits of the x-axis.
    y_lim : tuple (bottom, top)
        Limits of the y-axis.
    ax : axes, optional
        Axes where to draw the figure. If ax=None, axes will be created. 
    middle_value : float, default=0
         The x value to decide which labels go to the left and which go to the right.
    subset : bool, default= True
         If True, a subset of the dataset is used for the cluster center calculations.
    subset_size : int, default=500000
        Size of the subset of the dataset used for the cluster center calculations.
    rs : int, default= 42
         Random seed.
    fontsize: int, default=7
         fontsize for the labels.
    capitalize : bool, default = True
        If True, it will capitalize the labels.
    
    See Also
    --------
    find_cluster_center
    
    """
    
    assert x_lim[0] < x_lim[1], "xlim values are in the wrong order"
    assert y_lim[0] < y_lim[1], "ylim values are in the wrong order"
    
    if ax is None:
        fig, ax = plt.subplots()

    # calculate cluster centers
    center_cluster_coordinates = find_cluster_center(tsne, colors, legend, subset, subset_size, rs)
    
    # sort by x
    center_cluster_coordinates_left = center_cluster_coordinates[center_cluster_coordinates.x < middle_value].copy()
    center_cluster_coordinates_right = center_cluster_coordinates[center_cluster_coordinates.x >= middle_value].copy()

    # sort by y
    center_cluster_coordinates_left.sort_values(by = 'y', inplace=True, ascending = False)
    center_cluster_coordinates_right.sort_values(by = 'y', inplace=True, ascending = False)
    
    # sort by top & bottom
    y_lim_top_left = 25
    y_lim_bottom_left = -50
    
    y_lim_top_right = 0
    y_lim_bottom_right = -75 #-100

    
    center_cluster_coordinates_left_top = center_cluster_coordinates_left[center_cluster_coordinates_left.y > y_lim_top_left]
    center_cluster_coordinates_left_bottom = center_cluster_coordinates_left[center_cluster_coordinates_left.y <= y_lim_bottom_left]
    center_cluster_coordinates_right_top = center_cluster_coordinates_right[center_cluster_coordinates_right.y > y_lim_top_right]
    center_cluster_coordinates_right_bottom = center_cluster_coordinates_right[center_cluster_coordinates_right.y <= y_lim_bottom_right]
    
    sorted_labels_left_top = center_cluster_coordinates_left_top.index.tolist()
    sorted_labels_left_bottom = center_cluster_coordinates_left_bottom.index.tolist()
    sorted_labels_right_top = center_cluster_coordinates_right_top.index.tolist()
    sorted_labels_right_bottom = center_cluster_coordinates_right_bottom.index.tolist()

    sorted_colors_left_top = np.vectorize(legend.get)(sorted_labels_left_top)
    sorted_colors_left_bottom = np.vectorize(legend.get)(sorted_labels_left_bottom)
    sorted_colors_right_top = np.vectorize(legend.get)(sorted_labels_right_top)
    sorted_colors_right_bottom = np.vectorize(legend.get)(sorted_labels_right_bottom)
    
    
    if capitalize == True:
        sorted_labels_left_top = [elem.capitalize() for elem in sorted_labels_left_top]
        sorted_labels_left_bottom = [elem.capitalize() for elem in sorted_labels_left_bottom]
        sorted_labels_right_top = [elem.capitalize() for elem in sorted_labels_right_top]
        sorted_labels_right_bottom = [elem.capitalize() for elem in sorted_labels_right_bottom]


    # PLOT
    # left top 
    n_left_top=len(sorted_labels_left_top)
    x=x_lim[0]*np.ones(n_left_top)
    y=np.linspace(y_lim[1], y_lim_top_left, n_left_top)

    for i, colr in enumerate(sorted_colors_left_top):
        #manually defining y position of 'Sclerosis'
        if sorted_labels_left_top[i] == 'Sclerosis':
            y[i]=0
        
        # moving down the sensory/retina/olfactory labels
        if sorted_labels_left_top[i] == 'Sensory':
            y[i]-= 15
            
        if sorted_labels_left_top[i] == 'Retina':
            y[i]-= 10
            
        if sorted_labels_left_top[i] == 'Olfactory':
            y[i]-= 5
            
            
        if any( [colr=='black', colr=='#0000A6', colr=='#5A0007', colr=='#4A3B53', colr=='#1B4400',
                 colr=='#004D43', colr=='#013349', colr=='#000035', colr=='#300018', colr=='#001E09',
                 colr=='#372101', colr=='#6508ba', colr=='#6F0062'] ):            
            # white colored letters
            ax.text(x[i], y[i], sorted_labels_left_top[i], c='lightgrey', fontsize=fontsize, ha='right', bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
            ax.plot([x[i],center_cluster_coordinates_left_top.x[i]],[y[i],center_cluster_coordinates_left_top.y[i]], c=colr, linewidth=0.4)
        else:
            # black colored letters
            ax.text(x[i], y[i], sorted_labels_left_top[i], c='black', fontsize=fontsize, ha='right', bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
            ax.plot([x[i],center_cluster_coordinates_left_top.x[i]],[y[i],center_cluster_coordinates_left_top.y[i]], c=colr, linewidth=0.4)
    
    # left bottom 
    n_left_bottom=len(sorted_labels_left_bottom)
    x=x_lim[0]*np.ones(n_left_bottom)
    y=np.linspace(y_lim_bottom_left, y_lim[0], n_left_bottom)

    for i, colr in enumerate(sorted_colors_left_bottom):
        if any( [colr=='black', colr=='#0000A6', colr=='#5A0007', colr=='#4A3B53', colr=='#1B4400',
                 colr=='#004D43', colr=='#013349', colr=='#000035', colr=='#300018', colr=='#001E09',
                 colr=='#372101', colr=='#6508ba', colr=='#6F0062'] ):            
            # white colored letters
            ax.text(x[i], y[i], sorted_labels_left_bottom[i], c='lightgrey', fontsize=fontsize, ha='right', bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
            ax.plot([x[i],center_cluster_coordinates_left_bottom.x[i]],[y[i],center_cluster_coordinates_left_bottom.y[i]], c=colr, linewidth=0.4)
        else:
            # black colored letters
            ax.text(x[i], y[i], sorted_labels_left_bottom[i], c='black', fontsize=fontsize, ha='right', bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
            ax.plot([x[i],center_cluster_coordinates_left_bottom.x[i]],[y[i],center_cluster_coordinates_left_bottom.y[i]], c=colr, linewidth=0.4)

    
    # right top
    n_right_top=len(sorted_labels_right_top)
    x=x_lim[1]*np.ones(n_right_top)
    y=np.linspace(y_lim[1], y_lim_top_right, n_right_top)

    for i, colr in enumerate(sorted_colors_right_top):
        #manually defining y position of 'Alzheimer'
        if sorted_labels_right_top[i] == 'Alzheimer':
            y[i]=-25
            
        # color blanco
        if any( [colr=='black', colr=='#0000A6', colr=='#5A0007', colr=='#4A3B53', colr=='#1B4400',
                 colr=='#004D43', colr=='#013349', colr=='#000035', colr=='#300018', colr=='#001E09',
                 colr=='#372101', colr=='#6508ba', colr=='#6F0062'] ):
            ax.text(x[i], y[i], sorted_labels_right_top[i], c='lightgrey', fontsize=fontsize, ha='left', bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
            ax.plot([x[i],center_cluster_coordinates_right_top.x[i]],[y[i],center_cluster_coordinates_right_top.y[i]], c=colr, linewidth=0.4)
        else:
            ax.text(x[i], y[i], sorted_labels_right_top[i], c='black', fontsize=fontsize, ha='left', bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
            ax.plot([x[i],center_cluster_coordinates_right_top.x[i]],[y[i],center_cluster_coordinates_right_top.y[i]], c=colr, linewidth=0.4)
            
    # right bottom
    n_right_bottom=len(sorted_labels_right_bottom)
    x=x_lim[1]*np.ones(n_right_bottom)
    y=np.linspace(y_lim_bottom_right, y_lim[0], n_right_bottom)

    for i, colr in enumerate(sorted_colors_right_bottom):
        # color blanco
        if any( [colr=='black', colr=='#0000A6', colr=='#5A0007', colr=='#4A3B53', colr=='#1B4400',
                 colr=='#004D43', colr=='#013349', colr=='#000035', colr=='#300018', colr=='#001E09',
                 colr=='#372101', colr=='#6508ba', colr=='#6F0062'] ):
            ax.text(x[i], y[i], sorted_labels_right_bottom[i], c='lightgrey', fontsize=fontsize, ha='left', bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
            ax.plot([x[i],center_cluster_coordinates_right_bottom.x[i]],[y[i],center_cluster_coordinates_right_bottom.y[i]], c=colr, linewidth=0.4)
        else:
            ax.text(x[i], y[i], sorted_labels_right_bottom[i], c='black', fontsize=fontsize, ha='left', bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
            ax.plot([x[i],center_cluster_coordinates_right_bottom.x[i]],[y[i],center_cluster_coordinates_right_bottom.y[i]], c=colr, linewidth=0.4)
            
            
            
            
def plot_ml_label_tags(tsne, colors, legend, x_lim, y_lim, abbrv= None, ax=None, middle_value = 0, subset = True, subset_size = 500000, rs = 42, fontsize=7, capitalize=True):
    """Plots label tags and a line pointing to the embedding.
    The line from a label tag points to the location with higher points density of that specific label. The labels can be changed for display in the plot with `abbrv'.
    
    
    Parameters
    ----------
    tsne: array-like of shape (n_points,2)
        t-SNE coordinates.
    colors : array-like of shape (n_points,)
        Color values for the colormap.
    legend : dict
        Legend label-color.
    x_lim : tuple (left, right)
        Limits of the x-axis.
    y_lim : tuple (bottom, top)
        Limits of the y-axis.
    abbrv : dict, default=None
        Dictionary with the abbreviations of the labels for the plot.
    ax : axes, optional
        Axes where to draw the figure. If ax=None, axes will be created. 
    middle_value : float, default=0
         The x value to decide which labels go to the left and which go to the right.
    subset : bool, default= True
         If True, a subset of the dataset is used for the cluster center calculations.
    subset_size : int, default=500000
        Size of the subset of the dataset used for the cluster center calculations.
    rs : int, default= 42
         Random seed.
    fontsize: int, default=7
         fontsize for the labels.
    capitalize : bool, default = True
        If True, it will capitalize the labels.
    
    See Also
    --------
    find_cluster_center
    
    """
    
    assert x_lim[0] < x_lim[1], "xlim values are in the wrong order"
    assert y_lim[0] < y_lim[1], "ylim values are in the wrong order"
    
    if ax is None:
        fig, ax = plt.subplots()

    # calculate cluster centers
    center_cluster_coordinates = find_cluster_center(tsne, colors, legend, subset, subset_size, rs)
    
    # sort by x
    center_cluster_coordinates_left = center_cluster_coordinates[center_cluster_coordinates.x < middle_value].copy()
    center_cluster_coordinates_right = center_cluster_coordinates[center_cluster_coordinates.x >= middle_value].copy()

    # sort by y
    center_cluster_coordinates_left.sort_values(by = 'y', inplace=True, ascending = False)
    center_cluster_coordinates_right.sort_values(by = 'y', inplace=True, ascending = False)
    
    sorted_labels_left = center_cluster_coordinates_left.index.tolist()
    sorted_labels_right = center_cluster_coordinates_right.index.tolist()

    sorted_colors_left = np.vectorize(legend.get)(sorted_labels_left)
    sorted_colors_right = np.vectorize(legend.get)(sorted_labels_right)
    
    if abbrv is not None:
        capitalize = False
        
    if capitalize == True:
        sorted_labels_left = [elem.capitalize() for elem in sorted_labels_left]
        sorted_labels_right = [elem.capitalize() for elem in sorted_labels_right]


    # PLOT
    scale_factor_labels = 4.2
    
    # left
    if abbrv is not None:
        sorted_labels_left = np.vectorize(abbrv.get)(sorted_labels_left)
    labels_left_splited = [x.split() for x in sorted_labels_left]
    n_left=len(np.hstack(labels_left_splited))
    x=x_lim[0]*np.ones(n_left)
    y=np.linspace(y_lim[1], y_lim[0], n_left)
    
    n=0
    for i, colr in enumerate(sorted_colors_left):
        for j, label in enumerate(labels_left_splited[i]):
            #first word of label
            if j==0:
                if any( [colr=='black', colr=='#0000A6', colr=='#5A0007', colr=='#4A3B53', colr=='#1B4400',
                 colr=='#004D43', colr=='#013349', colr=='#000035', colr=='#300018', colr=='#001E09',
                 colr=='#372101', colr=='#6508ba'] ):            
                    # white colored letters
                    ax.text(x[n], y[n], label, c='lightgrey', fontsize=fontsize, ha='right', 
                            bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
                    ax.plot([x[n],center_cluster_coordinates_left.x[i]],[y[n],center_cluster_coordinates_left.y[i]], c=colr, linewidth=0.4, clip_on=False)
                else:
                    # black colored letters
                    ax.text(x[n], y[n], label, c='black', fontsize=fontsize, ha='right', 
                            bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
                    ax.plot([x[n],center_cluster_coordinates_left.x[i]],[y[n],center_cluster_coordinates_left.y[i]], c=colr, linewidth=0.4, clip_on=False)
                    
            # other words of label
            else:
                if any( [colr=='black', colr=='#0000A6', colr=='#5A0007', colr=='#4A3B53', colr=='#1B4400',
                 colr=='#004D43', colr=='#013349', colr=='#000035', colr=='#300018', colr=='#001E09',
                 colr=='#372101', colr=='#6508ba'] ):            
                    # white colored letters
                    ax.text(x[n], y[n-j]-fontsize*j*scale_factor_labels, label, c='lightgrey', fontsize=fontsize, ha='right', 
                            bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
                else:
                    # black colored letters
                    ax.text(x[n], y[n-j]-fontsize*j*scale_factor_labels, label, c='black', fontsize=fontsize, ha='right', 
                            bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
            n+=1

    # right
    if abbrv is not None:
        sorted_labels_right = np.vectorize(abbrv.get)(sorted_labels_right)
        
    labels_right_splited = [x.split() for x in sorted_labels_right]
    n_right=len(np.hstack(labels_right_splited))
    x=x_lim[1]*np.ones(n_right)
    y=np.linspace(y_lim[1], y_lim[0], n_right)
    
    n=0
    for i, colr in enumerate(sorted_colors_right):
        for j, label in enumerate(labels_right_splited[i]):
            #first word of label
            if j==0:
                if any( [colr=='black', colr=='#0000A6', colr=='#5A0007', colr=='#4A3B53', colr=='#1B4400',
                 colr=='#004D43', colr=='#013349', colr=='#000035', colr=='#300018', colr=='#001E09',
                 colr=='#372101', colr=='#6508ba'] ):            
                    # white colored letters
                    ax.text(x[n], y[n], label, c='lightgrey', fontsize=fontsize, ha='left', 
                            bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
                    ax.plot([x[n],center_cluster_coordinates_right.x[i]],[y[n],center_cluster_coordinates_right.y[i]], c=colr, linewidth=0.4, clip_on=False)
                else:
                    # black colored letters
                    ax.text(x[n], y[n], label, c='black', fontsize=fontsize, ha='left', 
                            bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
                    ax.plot([x[n],center_cluster_coordinates_right.x[i]],[y[n],center_cluster_coordinates_right.y[i]], c=colr, linewidth=0.4, clip_on=False)
                    
            # other words of label
            else:
                if any( [colr=='black', colr=='#0000A6', colr=='#5A0007', colr=='#4A3B53', colr=='#1B4400',
                 colr=='#004D43', colr=='#013349', colr=='#000035', colr=='#300018', colr=='#001E09',
                 colr=='#372101', colr=='#6508ba'] ):            
                    # white colored letters
                    ax.text(x[n], y[n-j]-fontsize*j*scale_factor_labels, label, c='lightgrey', fontsize=fontsize, ha='left', 
                            bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
                else:
                    # black colored letters
                    ax.text(x[n], y[n-j]-fontsize*j*scale_factor_labels, label, c='black', fontsize=fontsize, ha='left', 
                            bbox=dict(facecolor=colr,edgecolor='None', alpha=0.8, boxstyle='square', pad=0.05))
            n+=1

            
            
            
def plot_tsne_colors(tsne, colors, x_lim, y_lim, ax=None, plot_type=None, axis_on = False):
    """Plot t-SNE embedding with colors (by labels).
    
    Parameters
    ----------
    tsne: array-like
        t-SNE coordinates.
    colors : array-like
        Color values for the colormap.
    x_lim : tuple (left, right)
        Limits of the x-axis.
    y_lim : tuple (bottom, top)
        Limits of the y-axis.
    ax : axes, optional
        Axes where to draw the figure. If ax=None, axes will be created.
    plot_type : {None, 'subplot_2', 'subplot_3', 'subplot_3_grey', 'subregion', 'test'}, default=None
        Style of the plot, modifies dotsize and alpha.
    axis_on : bool, default=False
        If True, axis is shown in plot.
    
    """
        
    assert x_lim[0] < x_lim[1], "xlim values are in the wrong order."
    assert y_lim[0] < y_lim[1], "ylim values are in the wrong order."
    
    assert plot_type in [None, 'subplot_2', 'subplot_3', 'subplot_3_grey', 'subregion', 'test', 'pdf ML'], "Not valid `plot_type` value. Choose from [None, 'subplot_2', 'subplot_3', 'subplot_3_grey', 'subregion', 'test', 'pdf ML']."
    
    
    if ax is None:
        fig, ax = plt.subplots()
    
    s_grey = 0.1
    s_color = 0.5
    alpha_grey = 0.2
    alpha_color = 0.2
    
    if plot_type=='subplot_2':
        s_grey = 0.2
        s_color = 0.2
    
    if plot_type == 'subplot_3':
        s_grey = 0.1
        s_color = 0.1

    if plot_type == 'subplot_3_grey':
        s_grey = 0.05
        alpha_grey = 0.01
        s_color = 0.2
        alpha_color = 0.5
    
    if plot_type=='subregion':
        s_grey = 1
        s_color = 1
        alpha_grey = 0.6
        alpha_color = 0.7
        
    if plot_type=='test':
        s_grey = 2
        s_color = 2
        alpha_grey = 0.6
        alpha_color = 0.7
        
    if plot_type=='pdf ML':
        s_grey = 0.5
        alpha_grey = 0.02
        
        #s_grey = 0.2
        #alpha_grey = 0.2
        s_color = 0.2
        alpha_color = 0.5
    
    ax.scatter(tsne[:,0][colors=='lightgrey'], tsne[:,1][colors=='lightgrey'],
               s=s_grey, alpha=alpha_grey, c='lightgrey', marker= '.', linewidths=0, ec='None', rasterized=True) 
    ax.scatter(tsne[:,0][colors!='lightgrey'], tsne[:,1][colors!='lightgrey'],
               s=s_color, alpha=alpha_color, c=colors[colors!='lightgrey'], marker= '.', linewidths=0, ec='None', rasterized=True) 
    
    
    if plot_type=='subregion':
        ax.axis('scaled')
    else:
        ax.axis('equal')
        
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    if axis_on == False:
        ax.axis('off')
        
        
        
        
def plot_tsne_years(tsne, colors, x_lim, y_lim, ax=None, fontsize=7, plot_type=None, colorbar=True, colorbar_type=None, axis_on=False, rs = 42):
    """Plot t-SNE embedding with colors (by years).
    
    Parameters
    ----------
    tsne: array-like
        t-SNE coordinates.
    colors : array-like
        Color values for the colormap.
    x_lim : tuple (left, right)
        Limits of the x-axis.
    y_lim : tuple (bottom, top)
        Limits of the y-axis.
    ax : axes, optional
        Axes where to draw the figure. If ax=None, axes will be created.
    fontsize : int, default=7
        Fontsize for the years in the colorbar.
    plot_type : {None, 'subplot', 'subregion', 'test'}, default=None
        Style of the plot, modifies dotsize and alpha.
    colorbar : bool, default=True
        If True, colorbar will be plotted.
    colorbar_type : {None, 'neuroscience'}, default=None
        Style of the colorbar.
    axis_on : bool, default=False
        If True, axis is shown in plot.
    rs : int, default= 42
         Random seed for the reordering of points.
    
    """
    
    assert x_lim[0] < x_lim[1], "xlim values are in the wrong order."
    assert y_lim[0] < y_lim[1], "ylim values are in the wrong order."
    
    assert plot_type in [None, 'subplot', 'subregion', 'test'], "Not valid `plot_type` value. Choose from [None, 'subplot', 'subregion', 'test']."
    assert colorbar_type in [None, 'neuroscience'], "Not valid `colorbar_type` value. Choose from [None, 'neuroscience']."

    
    
    if ax is None:
        fig, ax = plt.subplots()
    
    s_color = 0.5
    alpha_color=0.2

    if plot_type == 'subplot':
        s_color = 0.2
        alpha_color=0.2
    
    if plot_type=='subregion':
        s_color = 0.5
        alpha_color = 0.7
        
    if plot_type=='test':
        s_color = 2
        alpha_color = 0.7
    
    np.random.seed(rs)
    reorder = np.random.permutation(tsne.shape[0])
    ax.scatter(tsne[reorder][:,0], tsne[reorder][:,1],s=s_color, c=colors[reorder],cmap='plasma', 
               alpha=alpha_color, marker= '.', linewidths=0, rasterized=True) 
    
    if plot_type=='subregion':
        ax.axis('scaled')
    else:
        ax.axis('equal')
        
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    if axis_on == False:
        ax.axis('off')
    
    if colorbar == True:
        if colorbar_type== 'neuroscience':
            heatmap = ax.scatter([], [], c=[], cmap='plasma')
            cbar = plt.colorbar(heatmap, ax=ax, shrink=0.1, location='left', anchor= (0,0), panchor= (0, 0), pad=-.3, aspect=10) 

        else:
            heatmap = ax.scatter([], [], c=[], cmap='plasma')
            cbar = plt.colorbar(heatmap, ax=ax, shrink=0.1, anchor= (0.5, 0), panchor= (0, 0.5), pad=-.13, aspect=10)
            # anchor second coordinate controls y-position and pad controls x-position

        cbar.set_alpha(1)
        cbar.ax.get_yaxis().set_ticks([0,1])
        cbar.ax.get_yaxis().set_ticklabels(['1970','2021'])
        cbar.ax.tick_params(labelsize=fontsize)
        
        
        

def plot_tsne_genders(tsne, colors, x_lim, y_lim, ax=None, plot_type=None, legend=True, axis_on = False, rs = 42):
    """Plot t-SNE embedding with colors (by genders).
    
    Parameters CHANGE
    ----------
    tsne: array-like
        t-SNE coordinates.
    colors : array-like
        Color values for the colormap.
    x_lim : tuple (left, right)
        Limits of the x-axis.
    y_lim : tuple (bottom, top)
        Limits of the y-axis.
    ax : axes, optional
        Axes where to draw the figure. If ax=None, axes will be created.
    plot_type : {None, 'subplot_2', 'subplot_3', 'subplot_3_grey', 'subregion', 'test'}, default=None
        Style of the plot, modifies dotsize and alpha.
    legend : bool, default=True
        If True, legend is shown.
    axis_on : bool, default=False
        If True, axis is shown in plot.
    rs : int, default= 42
         Random seed for the reordering of points.
    
    """
    
    assert x_lim[0] < x_lim[1], "xlim values are in the wrong order"
    assert y_lim[0] < y_lim[1], "ylim values are in the wrong order"
    
    assert plot_type in [None, 'subplot_2', 'subplot_3', 'subregion', 'test', 'zoom'], "Not valid `plot_type` value. Choose from [None, 'subplot_2', 'subplot_3', 'subregion', 'test', 'zoom']."
    
    if ax is None:
        fig, ax = plt.subplots()
    
    s = 0.5
    alpha=0.2

    if plot_type == 'subplot_2':
        s = 0.2
        alpha=0.2
        
    if plot_type == 'subplot_3':
        s=0.1 #0.05 v2
        alpha=0.2
        
    if plot_type == 'subregion':
        s = 2
        alpha=0.7
    
    if plot_type=='test':
        s = 10
        alpha= 1
        
    if plot_type=='zoom':
        s = 3
        alpha= 0.7
    
    np.random.seed(rs)
    reorder = np.random.permutation(tsne.shape[0])
    tsne_reordered = tsne[reorder]
    colors_reordered = colors[reorder]
    
    mask_pred_authors = (colors_reordered == 'tab:blue') | (colors_reordered == 'tab:orange')


    ax.scatter(tsne_reordered[colors_reordered == 'lightgrey'][:,0], tsne_reordered[colors_reordered == 'lightgrey'][:,1],
               s=s, c='lightgrey', alpha=alpha, marker= '.', linewidths=0, rasterized=True) 
    ax.scatter(tsne_reordered[colors_reordered == 'black'][:,0], tsne_reordered[colors_reordered == 'black'][:,1],
               s=s, c='black' , alpha=alpha, marker= '.', linewidths=0, rasterized=True) 
    ax.scatter(tsne_reordered[mask_pred_authors][:,0], tsne_reordered[mask_pred_authors][:,1],
               s=s, c=colors_reordered[mask_pred_authors] , alpha=alpha, marker= '.', linewidths=0, rasterized=True) 
    
    if legend == True:
        point1  = ax.scatter([], [], c='tab:orange', s=10, alpha=1 , label = 'female')
        point2  = ax.scatter([], [], c='tab:blue', s=10, alpha=1 , label = 'male')
        point3  = ax.scatter([], [], c='black', s=10, alpha=1 , label = 'unknown gender')
        point4  = ax.scatter([], [], c='lightgrey', s=10, alpha=1 , label = 'unknown name')

        ax.legend(handles=[point2, point1, point3, point4], loc = 'lower left', fontsize = 5, frameon=False,
                  borderpad = 0.2, handletextpad = 0, handlelength = 1, borderaxespad = 1.5) #-0.2 before
        
    ax.axis('equal')
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    if axis_on == False:
        ax.axis('off')
        
        
        
        
def plot_tsne_word(all_abstracts, word, tsne, x_lim, y_lim, ax=None, plot_type=None, title_on=False, axis_on = False, legend_on = False, verbose=True):
    """Plots t-SNE embedding with points having one given word in their abstract highlighted.
    It plots all points in grey, and papers that have that specific word/phrase in their abstract in black. 
    If more than one word is given, each of them will be plotted using colors from tab10 color palette.
    Take into account that if this happens, points will be plotted on top of each other instead of shuffled, so the amount of papers may be missleading.
    
    CURRENTLY NOT WORKING WHEN PASSING LIST OF WORDS INSTEAD OF SINGLE STR!
    
    Parameters
    ----------
    all_abstracts : pandas dataframe of str
        All texts (in this case abstracts).
    words : str or list of str
        Word/phrase or list with many words/phrases to be queried.
    tsne: array-like
        t-SNE coordinates.
    x_lim : tuple (left, right)
        Limits of the x-axis.
    y_lim : tuple (bottom, top)
        Limits of the y-axis.
    ax : axes, optional
        Axes where to draw the figure. If ax=None, axes will be created.
    plot_type : {None, 'subplot_2', 'subplot_3', 'subplot_3_grey', 'subregion', 'test'}, default=None
        Style of the plot, modifies dotsize and alpha.
    title_on : bool, default=False
        If True, adds the word being queried as title to the figure.
    axis_on : bool, default=False
        If True, axis is shown in plot.
    verbose : bool, default=True
        If True, prints the number of papers with that certain word and its variations in it .

    See Also
    --------
    exploration.find_mask_words
    
    """

    assert x_lim[0] < x_lim[1], "xlim values are in the wrong order"
    assert y_lim[0] < y_lim[1], "ylim values are in the wrong order"
    
    assert plot_type in [None, 'subplot_2', 'subplot_3', 'subplot_3_grey', 'subregion', 'test'], "Not valid `plot_type` value. Choose from [None, 'subplot_2', 'subplot_3', 'subplot_3_grey', 'subregion', 'test']."
    
    s_grey = 0.1
    s_color = 0.5
    alpha_grey = 0.2
    alpha_color = 0.5 #0.2
    
    if plot_type=='subplot_2':
        s_grey = 0.2
        s_color = 0.2
    
    if plot_type == 'subplot_3':
        s_grey = 0.1
        s_color = 0.1

    if plot_type == 'subplot_3_grey':
        s_grey = 0.05
        alpha_grey = 0.01
        s_color = 0.2
        alpha_color = 0.2 #0.5
    
    if plot_type=='subregion':
        s_grey = 1
        s_color = 1
        alpha_grey = 0.6
        alpha_color = 0.7
        
    if plot_type=='test':
        s_grey = 2
        s_color = 2
        alpha_grey = 0.6
        alpha_color = 0.7
    
    
    if ax is None:
        fig, ax = plt.subplots()

    if type(word) is str:
        mask = find_mask_words(all_abstracts, word, verbose=verbose)

        subregion=tsne[mask]
        ax.scatter(tsne[:,0], tsne[:,1], c = 'lightgrey', s=s_grey, alpha=alpha_grey, linewidths=0, rasterized=True)
        ax.scatter(subregion[:,0],subregion[:,1],s=s_color,c='black',alpha=alpha_color, marker='.', linewidths=0, rasterized=True)
        
        if title_on == True:
            #ax.set_title('"'+word+'"')
            ax.text(0.5,1, '"'+word+'"', transform=ax.transAxes, va='top', ha='center')
        
        
    elif type(word) is list:
        ax.scatter(tsne[:,0], tsne[:,1], c = 'lightgrey', s=s_grey, alpha=alpha_grey, linewidths=0, rasterized=True)
        
        for i, elem in enumerate(word):
            mask = find_mask_words(all_abstracts, elem, verbose=verbose)
            
            if verbose == True:
                print('---------------')
                
            subregion=tsne[mask]
            ax.scatter(subregion[:,0],subregion[:,1], c=np.matrix(plt.cm.tab10(i)), s=s_color,alpha=alpha_color, marker='.', linewidths=0, rasterized=True)
            ax.scatter([],[], c=np.matrix(plt.cm.tab10(i)), s=10, alpha=1, label='"'+elem+'"')
            
        if legend_on == True:
            ax.legend()
    
    ax.axis('equal')
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    
    if axis_on == False:
        ax.axis('off')
        
        
        

def plot_tsne_zoom(tsne, mask, x_lim, y_lim, ax=None, plot_type=None, title_on=False, axis_on=False, verbose=True):
    """Plots faster zoomed regions of t-SNE embedding.
    It plots all points in grey. One can pass an additional mask and it will color papers from the mask in black.

    Parameters
    ----------
    tsne: array-like
        t-SNE coordinates.
    mask : array of bool or None
        If given, colors points from that mask in black on top of the grey points.
    x_lim : tuple (left, right)
        Limits of the x-axis.
    y_lim : tuple (bottom, top)
        Limits of the y-axis.
    ax : axes, optional
        Axes where to draw the figure. If ax=None, axes will be created.
    plot_type : {None, 'zoom x2'}, default=None
        Style of the plot, modifies dotsize and alpha.
    title_on : bool, default=False
        If True, adds the word being queried as title to the figure.
    axis_on : bool, default=False
        If True, axis and grid are shown in plot.
    verbose : bool, default=True
        If True, prints the number of papers with that certain word and its variations in it .
    """

    assert x_lim[0] < x_lim[1], "xlim values are in the wrong order"
    assert y_lim[0] < y_lim[1], "ylim values are in the wrong order"

    assert plot_type in [
        None,
        "zoom x2",
    ], "Not valid `plot_type` value. Choose from [None, 'zoom x2']."

    s_grey = 3
    s_color = 3
    alpha_grey = 0.5
    alpha_color = 0.7

    if plot_type == "zoom x2":
        s_grey = 5
        s_color = 5
        alpha_grey = 0.5
        alpha_color = 0.7

    if ax is None:
        fig, ax = plt.subplots()


    mask_grey = (
        (tsne[:, 0] < x_lim[1])
        & (tsne[:, 0] > x_lim[0])
        & (tsne[:, 1] < y_lim[1])
        & (tsne[:, 1] > y_lim[0])
    )

    # plot
    ax.scatter(
        tsne[mask_grey, 0],
        tsne[mask_grey, 1],
        s=s_grey,
        c="lightgrey",
        alpha=alpha_grey,
        marker=".",
        # linewidths=0,
        ec="None",
        rasterized=True,
    )
    
    if mask is not None:
        mask_colors = (
            (tsne[:, 0] < x_lim[1])
            & (tsne[:, 0] > x_lim[0])
            & (tsne[:, 1] < y_lim[1])
            & (tsne[:, 1] > y_lim[0])
            & mask
        )

        ax.scatter(
            tsne[mask_colors, 0],
            tsne[mask_colors, 1],
            s=s_color,
            c="black",
            alpha=alpha_color,
            marker=".",
            # linewidths=0,
            ec="None",
            rasterized=True,
        )

    ax.axis("equal")
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])

    if axis_on == False:
        ax.axis("off")
    else:
        ax.grid()