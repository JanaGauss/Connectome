import numpy as np
import bct
import pandas as pd
from src.preprocessing.data_loader import flat_to_mat
from IPython.display import clear_output

# How to use:
# given a list of connectivity matrices just call the
# "get_graph_metrics" function on this list
# To get some explanations about the implemented
# graph metrics just call "explain_graph_metrics"
# without specifying any arguments
#


def get_graph_metrics(conns: list,
                      threshold: float = 0.65,
                      col_names: list = None,
                      use_abs: bool = False,
                      gms: tuple = ('Degrees',
                                    'Modularity',
                                    'Community Structure',
                                    'Clustering Coefficient',
                                    'Characteristic Path Length',
                                    'Node Betweenness',
                                    'Density',
                                    'Component Vectors',
                                    'Transitivity')) -> tuple:
    """
    - computes graph metrics for the given connectivity data
    - retransforms the connectivity matrices to pearson correlation before
      computing the adjacency matrices

    Args:
        conns: list of numpy arrays containing the connectivity data
        threshold: threshold of correlation to compute adjacency matrices
        col_names: colnames of the connectivity matrices in case those are reordered
            IMPORTANT: refers to the regions - e.g. "1", "4" instead of "1_4", "1_5"
        use_abs: whether absolute connectivity values should be used
        gms: graph metrics to be computed - can be any of:
                        ['Degrees', 'Modularity',
                        'Community Structure', 'Clustering Coefficient',
                        'Characteristic Path Length', 'Node Betweenness',
                        'Density', 'Component Vectors', 'Transitivity']

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
    stacked_conn = np.abs(np.tanh(np.stack(conns))) \
        if use_abs else np.tanh(np.stack(conns))
    adj_matrices = res["Adjacency Matrices"] = np.where(stacked_conn > threshold, 1, 0)
    regions = adj_matrices[0, :, :].shape[0]
    failed = {}

    for i in range(n):
        try:
            adj = adj_matrices[i, :, :]

            # community structure / modularity
            if "Modularity" in gms:
                if not any((adj != 0).flatten()):
                    res["Modularity"].append(0)
                    res["Community Structure"].append(np.zeros((regions,)))

                else:
                    community_struct = bct.modularity_und(adj)
                    res["Modularity"].append(community_struct[1])
                    res["Community Structure"].append(community_struct[0])

            # degrees
            if "Degrees" in gms:
                res["Degrees"].append(bct.degrees_und(adj))

            # clustering coef
            if "Clustering Coefficient" in gms:
                res["Clustering Coefficient"].append(bct.clustering_coef_bu(adj))

            # distance
            if "Characteristic Path Length" in gms:
                distance = bct.distance_bin(adj)
                min_dist = np.min(distance, axis=1)
                res["Characteristic Path Length"].append(np.array(np.mean(min_dist)))

            # node betweenness
            if "Node Betweenness" in gms:
                res["Node Betweenness"].append(bct.betweenness_bin(adj))

            # density
            if "Density" in gms:
                res["Density"].append(np.array(bct.density_und(adj)))

            # components of an undirected graph
            if "Component Vectors" in gms:
                graph_comp = np.array(bct.get_components(adj))
                res["Component Vectors"].append(graph_comp[0])

            # transitivity
            if "Transitivity" in gms:
                res["Transitivity"].append(0 if np.isnan(bct.transitivity_bu(adj))
                                           else bct.transitivity_bu(adj))

        except Exception as e:
            failed[i] = (e, adj_matrices[i, :, :])

    for key in res.keys():
        if key in ['Characteristic Path Length', 'Modularity', 'Transitivity']:
            res[key] = np.stack(res[key]).reshape(n, 1)
        else:
            res[key] = np.stack(res[key])

    if col_names is None:
        colnames = {
            "Degrees":
                ["degree_" + str(i + 1) for i in range(regions)],

            "Modularity":
                ["modularity"],

            "Community Structure":
                ["community_structure_" + str(i + 1) for i in range(regions)],

            "Clustering Coefficient":
                ["clustering_coefficient_" + str(i + 1) for i in range(regions)],

            "Characteristic Path Length":
                ["characteristic_path_length"],

            "Node Betweenness":
                ["node_betweenness_" + str(i + 1) for i in range(regions)],

            "Density":
                ["density", "vertices", "edges"],

            "Component Vectors":
                ["component_" + str(i + 1) for i in range(regions)],

            "Transitivity":
                ["transitivity"]
        }

    else:
        colnames = {
            "Degrees":
                ["degree_" + str(i) for i in col_names],

            "Modularity":
                ["modularity"],

            "Community Structure":
                ["community_structure_" + str(i) for i in col_names],

            "Clustering Coefficient":
                ["clustering_coefficient_" + str(i) for i in col_names],

            "Characteristic Path Length":
                ["characteristic_path_length"],

            "Node Betweenness":
                ["node_betweenness_" + str(i) for i in col_names],

            "Density":
                ["density", "vertices", "edges"],

            "Component Vectors":
                ["component_" + str(i) for i in col_names],

            "Transitivity":
                ["transitivity"]
        }

    dfs = []

    for key in res.keys():
        if key == "Component Vectors":
            # transforming component vectors into dummie variables
            # --> region x belongs to community y
            dummies = []
            df = pd.DataFrame(res[key], columns=colnames[key], index=range(n))
            for col in df.columns:
                d = pd.get_dummies(df[col])
                d.columns = [col + "_" + str(i) for i in d.columns]
                dummies.append(d)
            component_vectors_df = pd.concat(dummies, axis=1)
            dfs.append(component_vectors_df)

        if key != "Adjacency Matrices" and key != "Component Vectors":
            dfs.append(pd.DataFrame(res[key], columns=colnames[key], index=range(n)))

    return pd.concat(dfs, axis=1), failed


def is_conn_col(x: str):
    spl = x.split("_")
    if len(spl) < 2:
        return False
    else:
        try:
            res = (isinstance(int(spl[0]), int) and
                   isinstance(int(spl[1]), int))
        except:
            res = False
        if res:
            return True
        return False


def pd_to_arrays(data: pd.DataFrame,
                 cols: list) -> list:

    if cols is None:
        cols = [col for col in data.columns if is_conn_col(col)]
    conn_data = data.loc[:, cols].copy()
    return [flat_to_mat(np.array(row)) for ind, row in conn_data.iterrows()]


def get_gms_from_pd(data: pd.DataFrame,
                    regions: list = None,
                    cols: list = None,
                    **kwargs) -> pd.DataFrame:
    """

    Args:
        data: dataFrame containing the conn data
        regions: list of names of the regions of the conn matrix
        cols: list of columns of the DataFrame data which contain conn data
        **kwargs: anything that´s passed to "get_grap_metrics"

    Returns:
        DataFrame containing the computed graph metrics

    """

    arrays = pd_to_arrays(data, cols)
    return get_graph_metrics(conns=arrays,
                             #col_names=regions,
                             **kwargs)[0]


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


if __name__ == "__main__":
    # checking the is_conn_col() function
    test_names = ["1_2", "asf", "as_asd"]
    print([is_conn_col(x) for x in test_names])

    # checking the get_gms_from_pd function
    k = 8  # dim of the conn matrix
    obs = 10  # observations
    conn = pd.DataFrame(
        np.random.normal(
            loc=0.1,
            scale=1.2,
            size=int((k*(k-1)/2)*obs)).reshape(obs, int((k*(k-1)/2))),
            columns=[str(i) + "_" + str(j)
                     for i in range(k)
                     for j in range(i+1, k)])
    conn_data_list = [flat_to_mat(conn.iloc[i, :]) for i in range(obs)]
    print(conn)
    print(get_gms_from_pd(conn, conn.columns))
    print(get_graph_metrics(conn_data_list))
