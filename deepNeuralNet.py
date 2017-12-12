import tensorflow as tf
import numpy as np

#def getTfModel():
    #FEATURES = ['width', 'px', 'height', 'py', 'depth']

    #feature_columns = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]


    #
    # def input_fn(x_data, y_data):
    #     feature_cols = {k: tf.constant(x_data[:, i], shape=[len(x_data[:, i]), 1])
    #                     for i, k in enumerate(FEATURES)}
    #     if y_data is None:
    #         return feature_cols
    #
    #     labels = tf.constant(y_data)
    #     return feature_cols, labels
    #
    # return regressor, input_fn


def runDNN(Xtrain, Ytrain,Xtest):
    FEATURES = ['width', 'px', 'height', 'py', 'depth']

    feature_columns = [
        # "curb-weight" and "highway-mpg" are numeric columns.
        tf.feature_column.numeric_column(key="width"),
        tf.feature_column.numeric_column(key="px"),
        tf.feature_column.numeric_column(key="height"),
        tf.feature_column.numeric_column(key="py"),
        tf.feature_column.numeric_column(key="depth"),
    ]

    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns, hidden_units=[4, 3])

    def input_fn(x_data, y_data):
        feature_cols = {k: tf.constant(x_data[:, i], shape=[len(x_data[:, i]), 1]) for i, k in enumerate(FEATURES)}
        if y_data is None:
            return feature_cols

        labels = tf.constant(y_data)
        return feature_cols, labels

    #training
    regressor.fit(input_fn=lambda: input_fn(Xtrain, Ytrain), steps=2000)

    #testing
    yhat = regressor.predict(input_fn=lambda:input_fn(Xtest))

    return yhat