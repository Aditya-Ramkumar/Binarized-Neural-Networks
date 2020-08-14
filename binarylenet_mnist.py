import tensorflow as tf
import larq as lq
import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils

# Import MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten images
X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32')
X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

# Normalize pixel values 
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, 10) * 2 - 1

batch_size = 100
epochs = 10

# BN
e = 1e-6
momentum = 0.9

#LeNet5

# All quantized layers except the first will use the same options
kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip",
              kernel_initializer='glorot_uniform')

model = tf.keras.models.Sequential()

# For the first layer we only quantize the weights and not the input
# 6C5 - 2MP
model.add(lq.layers.QuantConv2D(6, 5, 
                                activation='relu',
                                padding='same',
                                input_shape=(32, 32, 1)))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.BatchNormalization(epsilon=e, momentum=momentum))

# 16C5 - 2MP
model.add(lq.layers.QuantConv2D(16, 5, padding='same', activation='relu', **kwargs))                           
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.BatchNormalization(epsilon=e, momentum=momentum))

model.add(tf.keras.layers.Flatten())

# FC120
model.add(lq.layers.QuantDense(120, activation='relu', **kwargs))
model.add(tf.keras.layers.BatchNormalization(epsilon=e, momentum=momentum))

# FC84
model.add(lq.layers.QuantDense(84, activation='relu', **kwargs))
model.add(tf.keras.layers.BatchNormalization(epsilon=e, momentum=momentum))

# Output layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))

lq.models.summary(model)

# Train

model.compile(tf.keras.optimizers.Adam(lr=0.01, decay=0.0001),
              loss='squared_hinge',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

test_loss, test_acc = model.evaluate(X_test, Y_test)

print(f"Test accuracy {test_acc * 100:.2f} %")
