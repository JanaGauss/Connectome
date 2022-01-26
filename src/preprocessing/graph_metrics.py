from scipy import stats
import numpy as np
import bct
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output

# How to use:
# given a list of connectivity matrices just call the "get_graph_metrics" function on this list
# currently not implemented to work for pd.DataFrames
# but the list of numpy arrays / conn matrices can be easily constructed using the "flat_to_mat" function
# of the dataloader module. Just apply this function on the DataFrame with the
# flattened conn data row-wise and obtain the list.
# To get some explanations about the implemented graph metrics just call "explain_graph_metrics"
# without specifying any arguments
#


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
            res[key] = np.stack(res[key]).reshape(n - len(failed.keys()), 1)
        else:
            res[key] = np.stack(res[key])

    if col_names is None:
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
            "Degrees": ["degree_" + str(i) for i in col_names],
            "Modularity": ["modularity"],
            "Community Structure": ["community_structure_" + str(i) for i in col_names],
            "Clustering Coefficient": ["clustering_coefficient_" + str(i) for i in col_names],
            "Characteristic Path Length": ["characteristic_path_length"],
            "Node Betweenness": ["node_betweenness_" + str(i) for i in col_names],
            "Density": ["density", "vertices", "edges"],
            "Component Vectors": ["component_" + str(i) for i in col_names],
            "Transitivity": ["transitivity"]
        }

    dfs = []

    for key in res.keys():
        if key != "Adjacency Matrices":
            dfs.append(pd.DataFrame(res[key], columns=colnames[key], index=range(n)))

    return pd.concat(dfs, axis=1)


def explain_graph_metrics() -> None:
    """
    computes graph metrics for the given connectivity data
    Must work for list of np.arrays or pd.DataFrames
    naming of columns is important afterwards!!
    """

    docs = {

        "Degrees": bct.degrees_und.__doc__,
        "Modularity": bct.modularity_und.__doc__,
        "Community Structure": bct.modularity_und.__doc__,
        "Clustering Coefficient": bct.clustering_coef_bu.__doc__,
        "Characteristic Path Length": bct.distance_bin.__doc__,
        "Node Betweenness": bct.betweenness_bin.__doc__,
        "Density": bct.density_und.__doc__,
        "Component Vectors": bct.get_components.__doc__,
        "Transitivity": bct.transitivity_bu.__doc__
    }
    explain = True

    while explain:
        print("So far implemented graph metrics:")
        print(list(docs.keys()))
        print("-" * 50 + "\n" * 2)
        check = input("Which one to explain?\t")

        if check not in docs.keys():
            print("selected graph metric not implemented / spelling error")
        else:
            print(docs[check])

        contin = input("Continue?(y)\t")
        clear_output(wait=True)  # only works well in a notebook environment
        explain = False if contin != "y" else True


