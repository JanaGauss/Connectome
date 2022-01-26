from scipy import stats
import numpy as np
import bct
import matplotlib.pyplot as plt
import pandas as pd


def get_graph_metrics(conns: list, threshold: float = 0.65, col_names: list = None,
                      use_abs: bool = False) -> pd.DataFrame:
    """
    - computes graph metrics for the given connectivity data
    - retransforms the connectivity matrices to pearson correlation before computing the adjacency matrices

    Args:
        conns: list of numpy arrays containing the connectivity data
        threshold: threshold of correlation to compute adjacency matrices
        col_names: colnames of the connectivity matrices in case those are reordered
        use_abs: boolean indicating whether to consider the absolute values
                of the correlation before computing the adjacency matrices

    Returns:
        pd.DataFrame containing the computed graph metrics

    """

    n = len(conns)
    res = {
        "Degrees": [],
        "Modularity": [],
        "Community Structure": [],
        "Clustering Coefficient": [],
        "Characteristic Path Length": [],
        "Node Betweenness": [],
        "Density": [],
        "Component Vectors": [],
        "Transitivity": []
    }
    stacked_conn = np.abs(np.tanh(np.stack(conns))) if use_abs else np.tanh(np.stack(conns))
    adj_matrices = res["Adjacency Matrices"] = np.where(stacked_conn > threshold, 1, 0)
    regions = adj_matrices[0, :, :].shape[0]
    failed = {}

    for i in range(n):
        try:
            # degrees
            adj = adj_matrices[i, :, :]
            res["Degrees"].append(bct.degrees_und(adj))

            # community structure / modularity
            community_struct = bct.modularity_und(adj)
            res["Modularity"].append(community_struct[1])
            res["Community Structure"].append(community_struct[0])

            # clustering coef
            res["Clustering Coefficient"].append(bct.clustering_coef_bu(adj))

            # distance
            distance = bct.distance_bin(adj)
            min_dist = np.min(distance, axis=1)
            res["Characteristic Path Length"].append(np.array(np.mean(min_dist)))

            # node betweenness
            res["Node Betweenness"].append(bct.betweenness_bin(adj))

            # density
            res["Density"].append(np.array(bct.density_und(adj)))

            # components of an undirected graph
            graph_comp = np.array(bct.get_components(adj))
            res["Component Vectors"].append(graph_comp[0])

            # transitivity
            res["Transitivity"].append(bct.transitivity_bu(adj))

        except Exception as e:
            failed[i] = (e, adj_matrices[i, :, :])

    for key in res.keys():
        if key in ['Characteristic Path Length', 'Modularity', 'Transitivity']:
            res[key] = np.stack(res[key]).reshape(n, 1)
        else:
            res[key] = np.stack(res[key])

    if col_names == None:
        colnames = {
            "Degrees": ["degree_" + str(i + 1) for i in range(regions)],
            "Modularity": ["modularity"],
            "Community Structure": ["community_structure_" + str(i + 1) for i in range(regions)],
            "Clustering Coefficient": ["clustering_coefficient_" + str(i + 1) for i in range(regions)],
            "Characteristic Path Length": ["characteristic_path_length"],
            "Node Betweenness": ["node_betweenness_" + str(i + 1) for i in range(regions)],
            "Density": ["density", "vertices", "edges"],
            "Component Vectors": ["component_" + str(i + 1) for i in range(regions)],
            "Transitivity": ["transitivity"]
        }
    else:
        colnames = {
            "Degrees": ["degree_" + str(i) for i in cols],
            "Modularity": ["modularity"],
            "Community Structure": ["community_structure_" + str(i) for i in cols],
            "Clustering Coefficient": ["clustering_coefficient_" + str(i) for i in cols],
            "Characteristic Path Length": ["characteristic_path_length"],
            "Node Betweenness": ["node_betweenness_" + str(i) for i in cols],
            "Density": ["density", "vertices", "edges"],
            "Component Vectors": ["component_" + str(i) for i in cols],
            "Transitivity": ["transitivity"]
        }

    dfs = []

    for key in res.keys():
        if key != "Adjacency Matrices":
            dfs.append(pd.DataFrame(res[key], columns=colnames[key], index=range(n)))

    return pd.concat(dfs, axis=1)


# all functions work for one 2d or one 3d array (multiple 2d arrays - 1st dim = number of conn matrices)
def adjacency_matrix(mat: np.ndarray, threshold=0.) -> np.ndarray:
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
        return adj_mat.sum(axis=1)
    else:
        return adj_mat.sum(axis=0)


def average_degree(degrees: np.ndarray) -> tuple:
    return degrees.mean()


def degree_distr(degrees: np.ndarray, plot=False, max_num=246) -> tuple:
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
            # we start the range in list compr at -1 as we want to get density values for all values
            # from 0 to 246 (max possible degree / connections)
            # starting from -1 gives use the edges for value 0
            return values, plt.hist(degrees, bins=bin_edges, density=True)[0]

        else:
            return values, np.histogram(degrees, bins=bin_edges, density=True)[0]


def degree_stats(degrees: np.ndarray) -> np.ndarray:
    if len(degrees.shape) == 2:
        return np.stack([degrees.mean(axis=1), np.std(degrees, axis=1),
                         np.median(degrees, axis=1), stats.mode(degrees, axis=1)[0].flatten()], axis=1)

    else:
        return np.array([degrees.mean(), np.std(degrees), np.median(degrees), stats.mode(degrees)[0][0]])
