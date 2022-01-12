#all functions work for one 2d or one 3d array (multiple 2d arrays - 1st dim = number of conn matrices)
def adjacency_matrix(mat: np.ndarray, threshold = 0.) -> np.ndarray:
    """
    function to compute an adjacency matrix based on a given connectivity matrix and a threshold.
    
    Args:
        mat: 
        threshold: 
    
    Returns: 
        an nd.array - the adjacency matrix
    
    Raises: 
    
    """
    
    return np.where(mat > threshold, 1, 0)


def degree(adj_mat: np.ndarray) -> np.ndarray:
    if len(adj_mat.shape) == 3:
        return adj_mat.sum(axis = 1)
    else:
        return adj_mat.sum(axis = 0)


def average_degree(degrees: np.ndarray) -> tuple:
    return degrees.mean()
                       
                  
def degree_distr(degrees: np.ndarray, plot=False, max_num=246) -> np.ndarray:
    """
    plotting is only supported if a 1D array is provided
    """
    bin_edges = [i + 0.5 for i in range(-1, max_num + 1)]
    values = np.arange(max_num + 1)
    
    
    if len(degrees.shape) == 2:
        hists = []
        for i in range(degrees.shape[0]):
            hists.append(np.histogram(degrees[i], bins=bin_edges, density=True)[0])
        return values, np.stack(hists) 
    
    else:
        if plot:
            #we start the range in list compr at -1 as we want to get density values for all values 
            #from 0 to 246 (max possible degree / connections)
            #starting from -1 gives use the edges for value 0            
            return values, plt.hist(degrees, bins=bin_edges, density=True)[0]
    
        else:
            return values, np.histogram(degrees, bins=bin_edges, density=True)[0]
    
    
def degree_stats(degrees: np.ndarray) -> np.ndarray:
    if len(degrees.shape) == 2:
        return np.stack([degrees.mean(axis=1), np.std(degrees, axis=1), 
                         np.median(degrees, axis=1), stats.mode(degrees, axis=1)[0].flatten()], axis=1)
    
    else:
        return np.array([degrees.mean(), np.std(degrees), np.median(degrees), stats.mode(degrees)[0][0]])