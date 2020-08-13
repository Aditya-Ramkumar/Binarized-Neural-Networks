# https://arxiv.org/abs/1602.02830


import tensorflow as tf
import larq as lq

from keras.datasets import cifar10
from keras.utils import np_utils

# Load CIFAR10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Resize
X_train = X_train.reshape((50000, 32, 32, 3)).astype('float32')
X_test = X_test.reshape((10000, 32, 32, 3)).astype('float32')

# Normalize
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'training samples')
print(X_test.shape[0], 'test samples')

# One hot encode
Y_train = np_utils.to_categorical(y_train, 10) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, 10) * 2 - 1

batch_size = 50
epochs = 20

# BN
e = 1e-6
momentum = 0.9

# Network

# All quantized layers except the first will use the same options
kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip",
              kernel_initializer='glorot_uniform',
              use_bias=False)

model = tf.keras.models.Sequential()

# For the first layer we only quantize the weights and not the input
# 2 x 128C3
model.add(lq.layers.QuantConv2D(128, 3, 
                                kernel_quantizer='ste_sign',
                                kernel_constraint='weight_clip',
                                kernel_initializer='glorot_uniform',
                                activation='relu',
                                use_bias=False,
                                input_shape=(32, 32, 3)))
model.add(tf.keras.layers.BatchNormalization(epsilon=e, momentum=momentum))


model.add(lq.layers.QuantConv2D(128, 3, activation='relu', **kwargs))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.BatchNormalization(epsilon=e, momentum=momentum))

# 2 x 256C3
model.add(lq.layers.QuantConv2D(256, 3, activation='relu',**kwargs))
model.add(tf.keras.layers.BatchNormalization(epsilon=e, momentum=momentum))

model.add(lq.layers.QuantConv2D(256, 3, activation='relu',**kwargs))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.BatchNormalization(epsilon=e, momentum=momentum))

# 2 x 512C3
model.add(lq.layers.QuantConv2D(512, 3, activation='relu',**kwargs))
model.add(tf.keras.layers.BatchNormalization(epsilon=e, momentum=momentum))

model.add(lq.layers.QuantConv2D(512, 3, activation='relu',**kwargs))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
model.add(tf.keras.layers.BatchNormalization(epsilon=e, momentum=momentum))

tf.keras.layers.Flatten()

# 2 x FC1024
model.add(lq.layers.QuantDense(1024, **kwargs))
model.add(tf.keras.layers.BatchNormalization(epsilon=e, momentum=momentum))

model.add(lq.layers.QuantDense(1024, **kwargs))
model.add(tf.keras.layers.BatchNormalization(epsilon=e, momentum=momentum))

# L2 SVM Output layer
model.add(tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2()))
model.add(tf.keras.layers.Activation('linear'))


lq.models.summary(model)

# Train
model.compile(tf.keras.optimizers.Adam(lr=0.01, decay=0.0001),
              loss='squared_hinge',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

test_loss, test_acc = model.evaluate(X_test, Y_test)

print(f"Test accuracy {test_acc * 100:.2f} %")