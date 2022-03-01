# TODO: Write as one function, model.fit, check for classification, ... implement validation set and early stopping

import numpy as np
import random

import keras
import tensorflow as tf
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, Flatten, Concatenate
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import StandardScaler
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

import src.preprocessing.data_loader as dtl
from src.preprocessing.reorder_matrices_regions import reorder_matrices_regions
from src.preprocessing.data_loader import flat_to_mat_aggregation
from src.models.E2E_conv import E2E_conv


def model_brainnet_cnn(X, y, aggregation=True, train_data=True, augmentation=False, scale=.07,
                       batch_size=32, epochs=400, patience = 10, learning_rate=0.001, opt=Adam(), validation_size = .1):

    # TODO: implement case for when there is no strucutral information
    # add all possible parameters of brainnet
    X_train, X_train_struc, y_train = preprocess_for_cnn(X,y, aggregation = aggregation, train_data = train_data, augmentation = augmentation, scale = scale)

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

    brainnetcnn = brain_net_cnn(input_dim_img=input_img_dim, input_dim_struc=input_struc_dim, output_dim=1, opt=opt)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    brainnetcnn.fit([train_x, train_x_struc], train_y, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=([val_x, val_x_struc], val_y), callbacks=[callback])

    return brainnetcnn


# scaler = StandardScaler()
def preprocess_for_cnn(X,y, aggregation=True, train_data=True, augmentation=False, scale=.07):
    # create target

    X_img_cols = []
    X_struc_cols= []
    for x in X.columns:
        if len(x.split("_"))>1 and x.split("_")[0].isdigit() and x.split("_")[1].isdigit():
            X_img_cols.append(x)
        else:
            X_struc_cols.append(x)

    # turn flat array to matrix
    if aggregation:
        n_c = 8
        n_train = len(X)
        X_train_2d = np.zeros(n_train * n_c * n_c).reshape(n_train, n_c, n_c)

        # turn array to matrix
        for i in range(n_train):
            X_train_2d[i] = flat_to_mat_aggregation(X[X_img_cols].iloc[i, :])

        stacked = np.stack(X_train_2d, axis=0)

    else:
        n_c = dtl.flat_to_mat(X[X_img_cols].iloc[0, :]).shape[0]
        n_train = len(X)
        X_train_2d = np.zeros(n_train * n_c * n_c).reshape(n_train, n_c, n_c)

        # turn array to matrix
        for i in range(n_train):
            X_train_2d[i] = dtl.flat_to_mat(X[X_img_cols].iloc[i, :])

        stacked = np.stack(reorder_matrices_regions(X_train_2d, network='yeo7'), axis=0)

    # reshape data
    X_img = stacked.reshape(stacked.shape[0], stacked.shape[1], stacked.shape[2], 1)
    if X[X_struc_cols].shape[1] != 0:
        X_struc = X[X_struc_cols].to_numpy().reshape(stacked.shape[0], 3)
    else:
        X_struc = X[X_struc_cols].to_numpy()
    # add gaussian noise
    if augmentation:
        X_img_aug, X_struc_aug, y_aug = data_augmentation_gaussian(X_img, X_struc, y, scale=scale)
        return X_img_aug, X_struc_aug, y_aug
    else:
        return X_img, X_struc, y


def brain_net_cnn(input_dim_img, input_dim_struc, output_dim: int = 1, kernel_regularizer=keras.regularizers.l2(0.0005),
                  kernel_initializer='he_uniform', dropout_rate=0.5, opt=Adam(), loss="binary_crossentropy"):
    # Source: https://kawahara.ca/convolutional-neural-networks-for-adjacency-matrices/
    if input_dim_struc != 0 :
        input_img = Input(shape=input_dim_img, name='input_img')

        x = E2E_conv(rank=2, filters=32, kernel_size=(2, input_dim_img[0]), kernel_regularizer=kernel_regularizer,
                     input_shape=(input_dim_img[0], input_dim_img[1], input_dim_img[2]),
                     activation=LeakyReLU(alpha=0.33), data_format="channels_last")(input_img)

        x = E2E_conv(rank=2, filters=32, kernel_size=(2, input_dim_img[0]), kernel_regularizer=kernel_regularizer,
                     input_shape=(input_dim_img[0], input_dim_img[1], input_dim_img[2]),
                     activation=LeakyReLU(alpha=0.33), data_format="channels_last")(x)

        x = Conv2D(filters=64, kernel_size=(1, input_dim_img[0]), strides=(1, 1),
                   padding='valid',
                   kernel_regularizer=kernel_regularizer,
                   kernel_initializer=kernel_initializer, activation=LeakyReLU(alpha=0.33), name='Edge-to-Node')(x)

        x = Conv2D(filters=64, kernel_size=(input_dim_img[1], 1), strides=(1, 1),
                   padding='valid',
                   kernel_regularizer=kernel_regularizer,
                   kernel_initializer=kernel_initializer, activation=LeakyReLU(alpha=0.33), name='Node-to-Graph')(x)

        x = Dropout(dropout_rate)(x)
        x = Flatten()(x)

        # add strutural data such as age, etc.
        input_struc = Input(shape=input_dim_struc, name='input_struc')

        x = Concatenate(axis=-1)([x, input_struc])

        x = Dense(128, kernel_initializer=kernel_initializer, activation=LeakyReLU(alpha=0.33))(x)
        x = Dropout(dropout_rate)(x)

        x = Dense(128, kernel_initializer=kernel_initializer, activation=LeakyReLU(alpha=0.33))(x)
        x = Dropout(dropout_rate)(x)

        out = Dense(output_dim, activation="sigmoid")(x)

        # compile model
        model = Model(inputs=[input_img, input_struc], outputs=out)

        model.compile(loss=loss,
                      optimizer=opt,
                      metrics=[tf.keras.metrics.BinaryAccuracy(),
                               tf.keras.metrics.FalseNegatives()])
    else:
        input_img = Input(shape=input_dim_img, name='input_img')

        x = E2E_conv(rank=2, filters=32, kernel_size=(2, input_dim_img[0]), kernel_regularizer=kernel_regularizer,
                     input_shape=(input_dim_img[0], input_dim_img[1], input_dim_img[2]),
                     activation=LeakyReLU(alpha=0.33), data_format="channels_last")(input_img)

        x = E2E_conv(rank=2, filters=32, kernel_size=(2, input_dim_img[0]), kernel_regularizer=kernel_regularizer,
                     input_shape=(input_dim_img[0], input_dim_img[1], input_dim_img[2]),
                     activation=LeakyReLU(alpha=0.33), data_format="channels_last")(x)

        x = Conv2D(filters=64, kernel_size=(1, input_dim_img[0]), strides=(1, 1),
                   padding='valid',
                   kernel_regularizer=kernel_regularizer,
                   kernel_initializer=kernel_initializer, activation=LeakyReLU(alpha=0.33), name='Edge-to-Node')(x)

        x = Conv2D(filters=64, kernel_size=(input_dim_img[1], 1), strides=(1, 1),
                   padding='valid',
                   kernel_regularizer=kernel_regularizer,
                   kernel_initializer=kernel_initializer, activation=LeakyReLU(alpha=0.33), name='Node-to-Graph')(x)

        x = Dropout(dropout_rate)(x)
        x = Flatten()(x)


        x = Dense(128, kernel_initializer=kernel_initializer, activation=LeakyReLU(alpha=0.33))(x)
        x = Dropout(dropout_rate)(x)

        x = Dense(128, kernel_initializer=kernel_initializer, activation=LeakyReLU(alpha=0.33))(x)
        x = Dropout(dropout_rate)(x)

        out = Dense(output_dim, activation="sigmoid")(x)

        # compile model
        model = Model(inputs=[input_img], outputs=out)

        model.compile(loss=loss,
                      optimizer=opt,
                      metrics=[tf.keras.metrics.BinaryAccuracy(),
                               tf.keras.metrics.FalseNegatives()])


    return model





def data_augmentation_gaussian(X, X_struc, y, scale=.07):
    X_noise = X.copy()
    X_struc_noise = X_struc.copy()
    y_noise = y.copy()

    for i in range(X.shape[0]):
        # add gaussian noise
        noise = np.random.normal(loc=0, scale=scale, size=(X.shape[1], X.shape[2], 1))
        X_noise[i] = X[i] + np.triu(noise, k=1).reshape(X.shape[1], X.shape[2], 1) + np.triu(noise, k=1).T.reshape(
            X.shape[1], X.shape[2], 1)

    return np.concatenate((X, X_noise), axis=0), np.concatenate((X_struc, X_struc_noise), axis=0), np.concatenate(
        (y, y_noise), axis=0)
