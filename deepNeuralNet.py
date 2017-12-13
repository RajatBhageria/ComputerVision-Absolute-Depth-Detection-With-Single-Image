import tensorflow as tf
import numpy as np

def runDNN(Xtrain, Ytrain,Xtest):

    #define the model features
    #FEATURES = ['height/width','depth']
    FEATURES = ['outputs']
    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

    #define the DNN regressor
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, hidden_units=[1,1])

    #define the input functions for training
    def get_input_fn(xData, yData=None, num_epochs=None, shuffle=True):
        return tf.estimator.inputs.numpy_input_fn(
            #x={FEATURES[i]: xData[:,i] for i in range(0,2)},
            x={FEATURES[0]: xData},
            y=yData,
            num_epochs=num_epochs,
            shuffle=shuffle)

    #training
    regressor.fit(input_fn=get_input_fn(Xtrain, Ytrain, None, False), steps=5000)

    #evaluation and getting loss
    ev = regressor.evaluate(input_fn=get_input_fn(Xtrain, Ytrain, 1, False))
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))

    #testing and predicting
    yhat = regressor.predict(input_fn=get_input_fn(Xtest, None, 1, False))
    predictions = np.array(list(p for p in yhat))
    return predictions