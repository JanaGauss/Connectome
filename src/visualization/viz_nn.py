from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear


from src.visualization.viz_utils import plot_feature_map

def model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = tf.keras.activations.linear

def score_function(output):
    # The `output` variable refers to the output of the model,
    return [i[0] for i in output]
    #return (output[0][0], output[1][0], output[2][0] )

def nn_feature_visualization(model, X, y, method = 'saliency', average = False):
    """
    Returns plots for different feature attribution methods: Vanilla Saliency, Grad Cam, Grad Cam Plus.

    Args:
        model: A fitted neural network model
        X:
        y:
        method: 'saliency', 'gradcam' or 'gradcamplus'
        average: Return the average or individual feature maps

    Returns:
        Returns a list feature attribution plots for the desired method
    """
    #Assert that X is valid input for model
    if len(X) != len(model.inputs):
        raise ValueError(
            f"The model has {len(model.inputs)} inputs, "
            f"but the number of X-inputs tensors you passed is {len(X)}.")

    for i, (x, tensor) in enumerate(zip(X, model.inputs)):
        if len(x.shape) != len(tensor.shape):
            raise ValueError(
                f"seed_input's shape is invalid. model-input index: {i},"
                f" model-input shape: {tensor.shape}, seed_input shape: {x.shape}.")

    #assert that y and X have the same length

    assert isinstance(method, str) & (method == "saliency" or method == "gradcam" or method == "gradcamplus"), "Invalid input method! Choose one of the following: 'saliency', 'gradcam' or 'gradcamplus'"
    assert isinstance(average, bool),   "invalid datatype. Choose boolean"
    #cast as float32

    #replace last layer with linear layer
    replace2linear = ReplaceToLinear()

    if method == 'saliency':
        cam = Saliency(model,
                       model_modifier=replace2linear,
                       clone=True)

        feature_map = cam(score_function,
                          X)

    elif method == 'gradcam':

        # Create Gradcam object
        cam = Gradcam(model,
                      model_modifier=replace2linear,
                      clone=True)

        # Generate heatmap with GradCAM
        feature_map = cam(score_function,
                          X,
                          penultimate_layer=-1)
        print(3)
    elif method == 'gradcamplus':
        cam = GradcamPlusPlus(model,
                              model_modifier=replace2linear,
                              clone=True)
        # Generate heatmap with GradCAM++
        feature_map = cam(score_function,
                          X,
                          penultimate_layer=-1)

    figs = []
    #create visualization

    if average:
        for label in [0,1]:

            y_ind = [i for i, x in enumerate(y) if x==label]

            if label ==1:
                title = "Has Alzheimer"
            else:
                title = "No Alzheimer"

            heatmap = feature_map[0][y_ind].mean(axis = 0)
            if heatmap.shape[0]==8:
                aggregated_network = True
            else:
                aggregated_network = False

            fig = plot_feature_map(heatmap, title, aggregated_network)

            figs.append(fig)

        return figs
    else:
        for j in range(len(y)):
            #define title
            if y[j] ==1:
                title = "Has Alzheimer"
            else:
                title = "No Alzheimer"

            heatmap = feature_map[0][j]
            #matrix size of 8 or 246
            if heatmap.shape[0]==8:
                aggregated_network = True
            else:
                aggregated_network = False

            fig = plot_feature_map(heatmap, title, aggregated_network)

            figs.append(fig)


        return figs