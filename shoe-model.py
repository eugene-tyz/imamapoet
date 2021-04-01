import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

#### LOAD Fashion MNIST dataset ####
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

plt.imshow(training_images[0])
## NORMALISE Values ##
training_images = training_images / 255.0
test_images = test_images / 255.0
# print(training_labels[0])
# print(training_images[0])

# SEQUENTIAL: That defines a SEQUENCE of layers in the neural network
# FLATTEN: Remember earlier where our images were a square, when you printed them out? Flatten just takes that square and turns it into a 1 dimensional set.
# DENSE: Adds a layer of neurons
# Each layer of neurons need an activation function to tell them what to do. There's lots of options, but just use these for now.
# RELU effectively means "If X>0 return X, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network.
# SOFTMAX takes a set of values, and effectively picks the biggest one, so, for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0] -- The goal is to save a lot of coding!
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(
                                        128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)
