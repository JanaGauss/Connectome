"""
Helper functions for visualization
"""
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from src.preprocessing.data_loader import flat_to_mat, flat_to_mat_aggregation
from src.preprocessing.reorder_matrices_regions import reorder_matrices_regions


def plot_feature_map(heatmap, title, aggregated_network=False, ordered=False, cmap='gist_heat_r', center_0=False):
    """
    Plots a heatmap of the connectivity matrix


    Args:
        heatmap: The image data
        title: A title for the plot
        aggregated_network: Boolean, whether the matrices were aggregated based on yeo7
        ordered: Boolean, whether to reorder the matrices based on the yeo7 network. (True is recommended when training
            with Brainnetome data. Only applicable to data based on the brainnetome atlas.
        cmap: Choice of colormap from matplotlib
        center_0: Boolean, whether to center the cmap around 0

    Returns:
        Connectivity matrix plot
    """
    assert isinstance(aggregated_network, bool), "invalid datatype for argument aggregated_network"
    assert isinstance(ordered, bool), "invalid datatype for argument ordered"
    assert isinstance(title, str), "invalid option, must be string"

    if aggregated_network:
        ticklabel = ["0", "1", "2", "3", "4", "5", "6", "7"]

        fig, ax = plt.subplots(figsize=(10, 10))

        if center_0:
            divnorm = colors.TwoSlopeNorm(vmin=heatmap.min(), vcenter=0., vmax=heatmap.max())
            plt.imshow(heatmap, cmap=cmap, norm=divnorm)
        else:
            plt.imshow(heatmap, cmap=cmap)

        plt.colorbar()
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
        ax.set_xticklabels(ticklabel)
        ax.set_yticklabels(ticklabel)

        plt.hlines([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], -0.5, 7.5, linewidth=2)
        plt.vlines([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], -0.5, 7.5, linewidth=2)
        plt.title(title)
        plt.tight_layout()
        plt.close()

        return fig
    else:
        # create plot
        fig, ax = plt.subplots(figsize=(10, 10))

        if center_0:
            divnorm = colors.TwoSlopeNorm(vmin=heatmap.min(), vcenter=0., vmax=heatmap.max())
            plt.imshow(heatmap, cmap=cmap, norm=divnorm)
        else:
            plt.imshow(heatmap, cmap=cmap)

        if ordered:
            # create borders and labels for plot
            ordered_roi, ordered_region = ordered_regions()
            ordered_region = list(map(int, ordered_region))

            hlines = []
            for i in range(len(ordered_region) - 1):
                if ordered_region[i] < ordered_region[i + 1]:
                    hlines.append(i)
            temp = hlines
            temp.append(245.5)
            temp.insert(0, 0)
            ticks = []
            for i in range(len(temp) - 1):
                ticks.append(np.round(np.mean([temp[i], temp[i + 1]])))

            ticklabel = ["0", "1", "2", "3", "4", "5", "6", "7"]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(ticklabel)
            ax.set_yticklabels(ticklabel)

            plt.hlines(hlines, 0, 245, linewidth=2)
            plt.vlines(hlines, 0, 245, linewidth=2)

        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.close()

        return fig


def ordered_regions() -> list:
    """
    return list of indexes from yeo 7
    """
    lab_to_yeo7 = {'1': '6', '2': '4', '3': '7', '4': '6', '5': '7', '6': '7', '7': '3', '8': '3', '9': '2', '10': '2',
                   '11': '7', '12': '6', '13': '7', '14': '7', '15': '4', '16': '6', '17': '6', '18': '6', '19': '6',
                   '20': '6', '21': '6', '22': '6', '23': '7', '24': '6', '25': '3', '26': '3', '27': '5', '28': '6',
                   '29': '6', '30': '3', '31': '6', '32': '6', '33': '7', '34': '7', '35': '7', '36': '6', '37': '4',
                   '38': '4', '39': '4', '40': '4', '41': '7', '42': '7', '43': '7', '44': '7', '45': '5', '46': '6',
                   '47': '5', '48': '5', '49': '5', '50': '5', '51': '7', '52': '7', '53': '2', '54': '2', '55': '3',
                   '56': '3', '57': '2', '58': '2', '59': '2', '60': '2', '61': '4', '62': '4', '63': '3', '64': '3',
                   '65': '4', '66': '2', '67': '2', '68': '2', '69': '5', '70': '5', '71': '2', '72': '2', '73': '2',
                   '74': '2', '75': '2', '76': '2', '77': '5', '78': '5', '79': '7', '80': '7', '81': '7', '82': '6',
                   '83': '7', '84': '7', '85': '3', '86': '3', '87': '7', '88': '7', '89': '5', '90': '5', '91': '3',
                   '92': '3', '93': '5', '94': '5', '95': '7', '96': '5', '97': '3', '98': '3', '99': '6', '100': '6',
                   '101': '5', '102': '5', '103': '5', '104': '5', '105': '1', '106': '1', '107': '3', '108': '1',
                   '109': '5', '110': '5', '111': '5', '112': '1', '113': '1', '114': '1', '115': '5', '116': '5',
                   '117': '5', '118': '5', '119': '1', '120': '1', '121': '7', '122': '7', '123': '4', '124': '4',
                   '125': '3', '126': '3', '127': '3', '128': '3', '129': '3', '130': '3', '131': '2', '132': '2',
                   '133': '3', '134': '3', '135': '1', '136': '1', '137': '6', '138': '6', '139': '3', '140': '3',
                   '141': '7', '142': '6', '143': '3', '144': '7', '145': '2', '146': '2', '147': '6', '148': '6',
                   '149': '2', '150': '3', '151': '1', '152': '1', '153': '7', '154': '7', '155': '2', '156': '2',
                   '157': '2', '158': '2', '159': '3', '160': '2', '161': '2', '162': '2', '163': '2', '164': '2',
                   '165': '0', '166': '6', '167': '4', '168': '4', '169': '4', '170': '4', '171': '2', '172': '2',
                   '173': '4', '174': '4', '175': '7', '176': '7', '177': '0', '178': '0', '179': '7', '180': '4',
                   '181': '7', '182': '1', '183': '4', '184': '4', '185': '4', '186': '4', '187': '7', '188': '7',
                   '189': '1', '190': '1', '191': '1', '192': '1', '193': '1', '194': '1', '195': '1', '196': '1',
                   '197': '1', '198': '1', '199': '1', '200': '1', '201': '3', '202': '1', '203': '1', '204': '1',
                   '205': '1', '206': '1', '207': '1', '208': '1', '209': '1', '210': '1', '211': '0', '212': '0',
                   '213': '0', '214': '0', '215': '0', '216': '0', '217': '0', '218': '0', '219': '0', '220': '0',
                   '221': '0', '222': '0', '223': '0', '224': '0', '225': '0', '226': '0', '227': '0', '228': '0',
                   '229': '0', '230': '0', '231': '0', '232': '0', '233': '0', '234': '0', '235': '0', '236': '0',
                   '237': '0', '238': '0', '239': '0', '240': '0', '241': '0', '242': '0', '243': '0', '244': '0',
                   '245': '0', '246': '0'}

    sorted_keys = list(dict(sorted(lab_to_yeo7.items(), key=lambda item: item[1])).keys())
    sorted_keys_int = list(map(int, sorted_keys))
    sorted_keys_list = [x - 1 for x in sorted_keys_int]
    ordered_region = list(dict(sorted(lab_to_yeo7.items(), key=lambda item: item[1])).values())
    ordered_region = list(map(int, ordered_region))
    return (sorted_keys_list, ordered_region)


def plot_coef_elastic_net(model, title="Elastic Net coefficients"):
    """
    plot coefficients of elastic net model

    Args:
      model: fitted elastic net model

    Returns:
      plot

    """

    assert model.__class__.__name__ in ['LogisticRegressionCV', 'ElasticNetCV']

    # extract indices of conn variables
    ind_conn_cols = []
    for x in range(len(model.feature_names_in_)):
        if len(model.feature_names_in_[x].split("_")) > 1 and model.feature_names_in_[x].split("_")[0].isdigit() and \
                model.feature_names_in_[x].split("_")[1].isdigit():
            ind_conn_cols.append(x)

    if model.__class__.__name__ == 'LogisticRegressionCV':
        coefs = model.coef_[0][ind_conn_cols]
    else:
        coefs = model.coef_[ind_conn_cols]  # structure of model.coef_ is different for regression and classification

    mat = flat_to_mat(coefs)

    # define if aggregated based on yeo7 or not depending on shape
    if mat.shape[0] == 8:
        aggregated = True
    else:
        aggregated = False

    if aggregated:
        plot_mat = flat_to_mat_aggregation(coefs)
        plot = plot_feature_map(plot_mat, title=title, aggregated_network=True, cmap='seismic', center_0=True)
    else:  # reorder by regions
        plot_mat = reorder_matrices_regions([mat], network='yeo7')[0]
        plot = plot_feature_map(plot_mat, title=title, ordered=True, aggregated_network=False, cmap='seismic', center_0=True)

    return plot


def plot_grouped_FI(df_importance, title="Grouped Permutation Feature Importance"):
    """
    plot results grouped feature importance (groups based on yeo7 network)

    Args:
      df_importance: pd.DataFrame with results from calculation grouped FI. First column contains regions, second column contains importance values

    Returns:
      plot

    """

    assert isinstance(df_importance, pd.DataFrame), "provided df_importance is no pd.DataFrame"

    # set values < 0 to 0
    df_importance.iloc[:, 1][df_importance.iloc[:, 1] < 0] = 0

    # reorder results
    order_regs = ['0_0', '0_1', '0_2', '0_3',
                  '0_4', '0_5', '0_6', '0_7', '1_1', '1_2', '1_3', '1_4', '1_5', '1_6',
                  '1_7', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7', '3_3', '3_4', '3_5',
                  '3_6', '3_7', '4_4', '4_5', '4_6', '4_7', '5_5', '5_6', '5_7', '6_6',
                  '6_7', '7_7']

    result = []
    for i in order_regs:
        res_i = df_importance.loc[df_importance['region'] == i][df_importance.columns[1]].values[
            0]  # reorder Feature Importance Values -> ordered like in order_regs
        result.append(res_i)

    plot_mat = flat_to_mat_aggregation(result)
    return plot_feature_map(plot_mat, title=title, aggregated_network=True)
