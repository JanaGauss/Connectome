from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import tensorflow as tf


from src.visualization.viz_utils import plot_feature_map

def model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = tf.keras.activations.linear

def score_function(output):
    # The `output` variable refers to the output of the model,
    return [i[0] for i in output]


def nn_feature_visualization(model, X, y, method='saliency', average=False, ordered=False):
    """
    Returns plots for different feature attribution methods: Vanilla Saliency and Smooth Saliency.
    If you want to view e.g. the first 3 visualizations and not the average, but have structural data, the input should
    follow  the format  X=[X_test[0][0:3],X_test[1][0:3]], y=y_test[0:3]


    Args:
        model: A fitted neural network model
        X: A test dataframe to visualize
        y: labels
        method: 'saliency' or 'saliency_smooth'
        average: Return the average or individual feature maps
        ordered: Boolean, whether to reorder the matrices based on the yeo7 network. (True is recommended when training
            with Brainnetome data. Only applicable to data based on the brainnetome atlas.

    Returns:
        Returns a list feature attribution plots for the desired method
    """
    # assert that y and X have the same length
    if type(X) == list:
        struc=True
        if X[0].shape[0] != len(y):
            raise ValueError(f"The input dimensions of X and y don't match")
        X = [tf.cast(X[0], dtype=tf.float32), tf.cast(X[1], dtype=tf.float32)]
    else:
        if X.shape[0] != len(y):
            raise ValueError(f"The input dimensions of X and y don't match")
        X = tf.cast(X, dtype=tf.float32)

    assert isinstance(method, str) & (
                method == "saliency" or  method == "saliency_smooth"), "Invalid input method! Choose one of the following: 'saliency', 'gradcam' or 'gradcamplus'"
    assert isinstance(average, bool), "invalid datatype. Choose boolean"

    # cast as float32

    # replace last layer with linear layer
    replace2linear = ReplaceToLinear()

    if method == 'saliency':
        cam = Saliency(model,
                       model_modifier=replace2linear,
                       clone=True)

        feature_map = cam(score_function,
                          X)

    elif method == 'saliency_smooth':

        # Create Saliency object
        cam = Saliency(model,
                       model_modifier=replace2linear,
                       clone=True)

        feature_map = cam(score_function,
                          X,
                          smooth_samples=20,  # The number of calculating gradients iterations.
                          smooth_noise=0.20)

    figs = []
    # create visualization
    if average:
        for label in [0, 1]:
            y_ind = [i for i, x in enumerate(y) if x == label]

            if label == 1:
                title = "Has Alzheimer"
            else:
                title = "No Alzheimer"

            if struc:
                heatmap = feature_map[0][y_ind].mean(axis=0)
            else:
                heatmap = feature_map[y_ind].mean(axis=0)

            if heatmap.shape[0] == 8:
                aggregated_network = True
            else:
                aggregated_network = False

            fig = plot_feature_map(heatmap, title, aggregated_network, ordered=ordered)

            figs.append(fig)

        return figs
    else:

        for j in range(len(y)):
            # define title
            if y[j] == 1:
                title = "Has Alzheimer"
            else:
                title = "No Alzheimer"

            if struc:
                heatmap = feature_map[0][j]
            else:
                heatmap = feature_map[j]

            # matrix size of 8 or 246
            if heatmap.shape[0] == 8:
                aggregated_network = True
            else:
                aggregated_network = False

            fig = plot_feature_map(heatmap, title, aggregated_network, ordered=ordered)

            figs.append(fig)

        return figs