"""This is a tensorflow tutorial that I have adapted for the CNN part of our project!"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  print("printing to mark the start of an iteration")
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 480, 640, 3])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 8]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=2,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 480, 640, 8]
  # Output Tensor Shape: [batch_size, 240, 320, 8]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=4)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 240, 320, 8]
  # Output Tensor Shape: [batch_size, 240, 320, 16]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=4,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 240, 320, 16]
  # Output Tensor Shape: [batch_size, 120, 160, 16]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 120, 160, 16]
  # Output Tensor Shape: [batch_size, 120 * 160 * 16]
  pool2_flat = tf.reshape(pool2, [-1, 60 * 80 * 4])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 120 * 160 * 16]
  # Output Tensor Shape: [batch_size, 120*160]
  dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 19200]
  # Output Tensor Shape: [batch_size, 19200]
  logits = tf.layers.dense(inputs=dropout, units=307200)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": logits
      #"classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      #"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    print(predictions)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10) #honxestly this depth should be 10 but i dont care
  print('i am printing here right before calculating losses')
  #print(tf.cast(labels, tf.int32))
  loss = tf.reduce_sum(tf.square(labels - logits))

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  print(predictions["classes"])
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  data = np.load('data/nyu_dataset_images.npy')
  labels_data = np.load('data/nyu_dataset_depths.npy')
  train_data = np.array([data[:, :, :, i] for i in range(20)], dtype='float32')
  eval_data = np.array([data[:, :, :, i] for i in range(20, 30)], dtype='float32')
  # train_labels = np.array([labels_data[:,:,i] for i in range(20)])
  # eval_labels = np.array([labels_data[:,:,i] for i in range(20,30)])
  train_labels = np.array([np.array(labels_data[:, :, i]).flatten() for i in range(20)])  # use these for flattened labels
  eval_labels = np.array([np.array(labels_data[:, :, i]).flatten() for i in range(20, 30)])

  # Create the Estimator
  depth_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  #tensors_to_log = {"probabilities": "softmax_tensor"}
  #logging_hook = tf.train.LoggingTensorHook(
      #tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=1,
      num_epochs=None,
      shuffle=True)
  depth_classifier.train(
      input_fn=train_input_fn,
      steps=2)

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = depth_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
