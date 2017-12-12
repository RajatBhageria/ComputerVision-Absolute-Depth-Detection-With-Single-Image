import tensorflow as tf
import numpy as np

def runDNN(Xtrain, Ytrain,Xtest):

    FEATURES = ['height/width', 'px/py', 'depth']

    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, hidden_units=[2, 2])
    def get_input_fn(Xtrain, Ytrain=None, num_epochs=None, shuffle=True):
        return tf.estimator.inputs.numpy_input_fn(
            x={FEATURES[i]: Xtrain[:,i] for i in range(0,3)},
            y=Ytrain,
            num_epochs=num_epochs,
            shuffle=shuffle)

    #training
    regressor.fit(input_fn=get_input_fn(Xtrain, Ytrain, None, False), steps=5000)

    #evaluation
    ev = regressor.evaluate(
        input_fn=get_input_fn(Xtrain, Ytrain, 1, False))
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))

    #testing
    yhat = regressor.predict(input_fn=get_input_fn(Xtest, None, 1, False))
    predictions = np.array(list(p for p in yhat))
    return predictions