import tensorflow as tf
import numpy as np
from tensorflow import keras

#### CREATE NEURAL NETWORK ####
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#### COMPLIE NEURAL NETWORK ####
model.compile(optimizer='sgd', loss='mean_squared_error')

#### DATA ####
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

#### TRAIN NEURAL NETWORK ON DATA ####
#### epochs is the number of repetitions ####
model.fit(xs, ys, epochs=141)

#### method to have it figure out the Y for a previously unknown X ####
print(model.predict([10.0]))
