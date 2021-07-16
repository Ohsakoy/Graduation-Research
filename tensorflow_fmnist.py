import time
import math
from traceback import format_stack
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import cProfile
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import  Adam, RMSprop

MAX_EPOCHE = 100

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

#train_images = train_images[..., None]
#test_images = test_images[..., None]

#train_images = train_images / np.float32(255)
#test_images = test_images / np.float32(255)


dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(train_images[..., tf.newaxis]/255, tf.float32),
     tf.cast(train_labels, tf.int64)))
train_dataset = dataset.shuffle(1000).batch(64)


test_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(test_images[..., tf.newaxis]/255, tf.float32),
     tf.cast(test_labels, tf.int64)))
test_dataset = test_dataset.batch(64)


for images, labels in train_dataset.take(1):
  print(images.shape, labels.shape)

for t_images, t_labels in test_dataset.take(1):
  print(t_images.shape, t_labels.shape)

# モデルを構築する


def create_model():
  mnist_model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='tanh'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, activation='tanh'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='tanh'),
      tf.keras.layers.Dense(10)
  ])

  return mnist_model


mnist_model = create_model()



optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_history = []

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')

train_loss_history0 = []
train_accuracy_history0 = []
test_loss_history0 = []
test_accuracy_history0 = []


def train_step(images, labels):
  with tf.GradientTape() as tape:
    logits = mnist_model(images, training=True)
    
    loss_value = loss_object(labels, logits)
    train_accuracy.update_state(labels, logits)

  loss_history.append(loss_value.numpy().mean())
  grads = tape.gradient(loss_value, mnist_model.trainable_variables)
  optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))
  return loss_value


def test_step(t_images, t_labels):

  logits = mnist_model(t_images, training=False)
  t_loss = loss_object(t_labels, logits)

  test_loss(t_loss)
  test_accuracy.update_state(t_labels, logits)

  #return t_loss


accLossHistory_train = np.zeros(MAX_EPOCHE)


start = time.time()

for epoch in range(MAX_EPOCHE):
    total_loss_value = 0.0
    train_loss = 0.0
    for (batchId, (images, labels)) in enumerate(train_dataset):
      loss_value = train_step(images, labels)
      total_loss_value += loss_value

    batchId += 1
    train_loss = total_loss_value / batchId
    accLossHistory_train[epoch] = train_loss

    for t_images, t_labels in (test_dataset):
      test_step(t_images, t_labels)

    train_loss_history0.append(train_loss)
    train_accuracy_history0.append(train_accuracy.result())
    test_loss_history0.append(test_loss.result())
    test_accuracy_history0.append(test_accuracy.result())
    template = (
        "Epoch {}, Train Loss: {}, Train Accuracy: {},Test Loss: {}, Test Accuracy: {}")
    print(template.format(epoch+1, train_loss,
          train_accuracy.result(), test_loss.result(), test_accuracy.result()))

    with open('tens_model1.txt', 'a') as f:
      print('epoch %d, train_loss: %.4f train_acc: %.4f test_loss: %.4f test_acc: %.4f' %
            (epoch+1, train_loss, train_accuracy.result(), test_loss.result(), test_accuracy.result()), file=f)

    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()


end = time.time() - start
print(end)
with open('time.txt', 'a') as t:
    print(end, file=t)
