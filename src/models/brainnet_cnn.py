import numpy as np
import random

import keras
import tensorflow as tf
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, Flatten, Concatenate, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import StandardScaler
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

import src.preprocessing.data_loader as dtl
from src.preprocessing.reorder_matrices_regions import reorder_matrices_regions
from src.preprocessing.data_loader import flat_to_mat_aggregation
from src.models.E2E_conv import E2E_conv


def model_brainnet_cnn(X, y, aggregation=True, augmentation=False, scale=.04, augm_fact = 5000,
                       batch_size=32, epochs=400, patience = 5, validation_size = .25,
                       E2E_filter: int = 32, E2N_filter: int = 48, N2G_tiler: int = 64,
                       dense_filter: int = 64, dropout_rate=0.5,
                       kernel_regularizer=keras.regularizers.l2(0.01),
                       kernel_initializer='he_uniform', optimizer=Adam(), activation='relu',
                       loss="binary_crossentropy"
                       ):


    # add all possible parameters of brainnet
    X_train, X_train_struc, y_train = preprocess_for_cnn(X,y, aggregation = aggregation, augmentation = augmentation, scale = scale, augmentation_factor=augm_fact)

    #create validation set
    # Train val split
    shuffled_indices = list(range(len(X_train)))
    random.shuffle(shuffled_indices)

    val_ind = np.round(len(X_train) * validation_size)
    val_idxs = shuffled_indices[:val_ind]
    train_idxs = shuffled_indices[val_ind:]

    #validation set
    val_x = X_train[val_idxs]
    val_x_struc = X_train_struc[val_idxs]
    val_y = y_train[val_idxs]
    #train set
    train_x = X_train[train_idxs]
    train_x_struc = X_train_struc[train_idxs]
    train_y = y_train[train_idxs]

    #mode input dimension
    input_img_dim = (X_train.shape[1], X_train.shape[2], 1)
    input_struc_dim = (X_train_struc.shape[1])

    brainnetcnn = brain_net_cnn(input_dim_img=input_img_dim, input_dim_struc=input_struc_dim, output_dim=1,
                                E2E_filter = E2E_filter, E2N_filter = E2N_filter, N2G_tiler = N2G_tiler,
                                dense_filter = dense_filter, dropout_rate=dropout_rate,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer, opt=optimizer, activation=activation,
                                loss=loss)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    brainnetcnn.fit([train_x, train_x_struc], train_y, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=([val_x, val_x_struc], val_y), callbacks=[callback])

    return brainnetcnn



def preprocess_for_cnn(X,y, aggregation=True, augmentation=False, scale=.07, augmentation_factor = 5):
    # create target

    X_img_cols = []
    X_struc_cols= []
    for x in X.columns:
        if len(x.split("_"))>1 and x.split("_")[0].isdigit() and x.split("_")[1].isdigit():
            X_img_cols.append(x)
        else:
            X_struc_cols.append(x)

    if augmentation:
        print("Starting Data Augmentation")
        X_img_aug, X_struc_aug, y_aug = augmented_data(X, y, X_img_cols, X_struc_cols, sd=scale, augm_fact = augmentation_factor)
        #merging augmented data with input data
        X_img = pd.concat([X[X_img_cols], X_img_aug])
        X_struc = pd.concat([X[X_struc_cols], X_struc_aug])
        y = np.concatenate([np.array(y), y_aug], axis = 0)
    else:
        X_img = X[X_img_cols]
        X_struc = X[X_struc_cols]

    print("Turning flat array to matrix")
    # turn flat array to matrix
    if aggregation:
        n_c = 8
        n_train = len(X_img)
        X_train_2d = np.zeros(n_train * n_c * n_c).reshape(n_train, n_c, n_c)

        # turn array to matrix
        for i in range(n_train):
            X_train_2d[i] = flat_to_mat_aggregation(X_img.iloc[i, :])

        stacked = np.stack(X_train_2d, axis=0)

    else:
        n_c = dtl.flat_to_mat(X_img.iloc[0, :]).shape[0]
        n_train = len(X_img)
        X_train_2d = np.zeros(n_train * n_c * n_c).reshape(n_train, n_c, n_c)

        # turn array to matrix
        for i in range(n_train):
            X_train_2d[i] = dtl.flat_to_mat(X_img.iloc[i, :])

        stacked = np.stack(reorder_matrices_regions(X_train_2d, network='yeo7'), axis=0)

    # reshape data
    X_img = stacked.reshape(stacked.shape[0], stacked.shape[1], stacked.shape[2], 1)
    if X_struc.shape[1] != 0:
          X_struc = X_struc.to_numpy().reshape(stacked.shape[0], 3)
    else:
        X_struc = X_struc.to_numpy()

    return X_img, X_struc, y


def brain_net_cnn(input_dim_img, input_dim_struc, output_dim: int = 1,
                  E2E_filter: int = 32, E2N_filter: int = 48, N2G_tiler: int = 64,
                  dense_filter: int = 64, dropout_rate=0.5,
                  kernel_regularizer=keras.regularizers.l2(0.01),
                  kernel_initializer='he_uniform', opt=Adam(), activation='relu',
                  loss="binary_crossentropy"):
    # Source: https://kawahara.ca/convolutional-neural-networks-for-adjacency-matrices/
    input_img = Input(shape=input_dim_img, name='input_img')

    x = E2E_conv(rank=2, filters=E2E_filter, kernel_size=(2, input_dim_img[0]), kernel_regularizer=kernel_regularizer,
                 input_shape=(input_dim_img[0], input_dim_img[1], input_dim_img[2]),
                 activation=activation, data_format="channels_last")(input_img)
    x = BatchNormalization()(x)

    x = E2E_conv(rank=2, filters=E2E_filter, kernel_size=(2, input_dim_img[0]), kernel_regularizer=kernel_regularizer,
                 input_shape=(input_dim_img[0], input_dim_img[1], input_dim_img[2]),
                 activation=activation, data_format="channels_last")(x)

    x = BatchNormalization()(x)
    x = Conv2D(filters=E2N_filter, kernel_size=(1, input_dim_img[0]), strides=(1, 1),
               padding='valid',
               kernel_regularizer=kernel_regularizer,
               kernel_initializer=kernel_initializer, activation=activation, name='Edge-to-Node')(x)

    x = BatchNormalization()(x)
    x = Conv2D(filters=N2G_tiler, kernel_size=(input_dim_img[1], 1), strides=(1, 1),
               padding='valid',
               kernel_regularizer=kernel_regularizer,
               kernel_initializer=kernel_initializer, activation=activation, name='Node-to-Graph')(x)

    x = Dropout(dropout_rate)(x)
    x = Flatten()(x)

    # add strutural data such as age, etc.
    input_struc = Input(shape=input_dim_struc, name='input_struc')

    x = Concatenate(axis=-1)([x, input_struc])

    x = Dense(dense_filter, kernel_initializer=kernel_initializer, activation=activation)(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(dense_filter,kernel_initializer=kernel_initializer, activation=activation)(x)
    x = Dropout(dropout_rate)(x)

    out = Dense(output_dim, activation="sigmoid")(x)

    # compile model
    model = Model(inputs=[input_img, input_struc], outputs=out, name='BrainNetCNN')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.FalseNegatives()])

    return model


def augmented_data(x: pd.DataFrame, y: pd.Series, X_img_cols: list, X_struc_cols: list, sd=0.17, augm_fact=5):
    x_aug = np.array(x[X_img_cols].copy())
    noise = np.random.normal(
        scale=sd,
        size=x_aug.shape[0] * x_aug.shape[1] * augm_fact).reshape(
        x_aug.shape[0] * augm_fact, x_aug.shape[1])
    x_aug = np.vstack([x_aug] * augm_fact) + noise
    x_struc_aug = np.vstack([np.array(x[X_struc_cols])] * augm_fact)
    y_aug = np.hstack([np.array(y)] * augm_fact)

    return pd.DataFrame(x_aug, columns=X_img_cols), pd.DataFrame(x_struc_aug, columns=X_struc_cols), y_aug