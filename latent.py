import numpy as np
from tensorflow import keras
# This is the implemante to build the autoencoder part

class Autoencoder(keras.Model):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = keras.Sequential([
            keras.layers.Dense(150, activation='tanh'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(100, activation='tanh'),

            keras.layers.Dense(50, activation='tanh'),

            keras.layers.Dense(25, activation='tanh'),

        ])
        self.decoder = keras.Sequential([
            keras.layers.Dense(50, activation='tanh'),
            keras.layers.Dense(100, activation='tanh'),
            keras.layers.Dense(150, activation='tanh'),
            keras.layers.Dense(input_dim, activation='sigmoid'),
        ])

    def call(self, x):
        code = self.encoder(x)

        r = self.decoder(code)
        return r

    def get_reconstruction_error(self, x):
        r = self.predict(x)
        return keras.metrics.mean_squared_error(x, r)

    def predict_class(self, x, threshold):
        reconstruction_error = self.get_reconstruction_error(x)
        return np.where(reconstruction_error <= threshold, 0, 1)

    def get_latent_representations(self, x):
        return self.encoder(x).numpy()



