# -*- coding: utf-8 -*-
"""binarynet_mnist_mlp

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1--um3-InOgQt8AlGLN_-jNt6GGUF9vz9
"""

import tensorflow as tf
import larq as lq

import keras.backend as K
from keras.datasets import mnist
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler

batch_size = 100
epochs = 20

# network
num_units = 4096
hidden_layers = 3
use_bias = False

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BN
e = 1e-6
momentum = 0.9

# dropout
drop_hidden = 0.5

# Import MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten images
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Normalize pixel values 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, 10) * 2 - 1

# All quantized layers except the first will use the same options
kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip",
              kernel_initializer='glorot_uniform')

model = tf.keras.models.Sequential()

# In the first layer we only quantize the weights and not the input
model.add(lq.layers.QuantDense(num_units, 
                               kernel_quantizer="ste_sign",
                               kernel_constraint="weight_clip",
                               use_bias=use_bias,
                               input_shape=(784,), 
                               name='dense{}'.format(1)))
model.add(tf.keras.layers.BatchNormalization(epsilon=e, momentum=momentum, name='bn{}'.format(1)))
model.add(tf.keras.layers.Dropout(drop_hidden, name='drop{}'.format(1)))

for i in range(1, hidden_layers):
  model.add(lq.layers.QuantDense(num_units, use_bias=True, name='dense{}'.format(i+1), **kwargs))
  model.add(tf.keras.layers.BatchNormalization(epsilon=e, momentum=momentum, name='bn{}'.format(i+1)))
  model.add(tf.keras.layers.Dropout(drop_hidden, name='drop{}'.format(i+1)))

# L2 SVM Output layer
model.add(tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2()))
model.add(tf.keras.layers.Activation('linear'))


lq.models.summary(model)

model.compile(tf.keras.optimizers.Adam(lr=0.01, decay=0.0001),
              loss='squared_hinge',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

test_loss, test_acc = model.evaluate(X_test, Y_test)

print(f"Test accuracy {test_acc * 100:.2f} %")