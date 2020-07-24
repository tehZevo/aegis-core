from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
import numpy as np

from aegis_core.print_node import PrintNode
from aegis_core.keras_dataset_node import KerasDataset
from aegis_core.utils import start_nodes, build_vae
from aegis_core.vae import VAENode

input_shape = [28, 28, 1]
latent_size = 2
acti = "tanh"

encoder = Sequential([
  Flatten(input_shape=input_shape),
  Dense(64, activation=acti),
  Dense(32, activation=acti),
])

decoder = Sequential([
  Dense(32, input_shape=[latent_size], activation=acti),
  Dense(64, activation=acti),
  Dense(np.prod(input_shape), activation="sigmoid"),
  Reshape(input_shape)
])

vae, encoder, decoder = build_vae(encoder, decoder, input_shape, latent_size)
vae.summary()
encoder.summary()
decoder.summary()

start_nodes([
  KerasDataset(12400, dataset="mnist", steps=100),
  VAENode(12401, vae, encoder, decoder, input_url="12400").set_niceness(0),
  PrintNode(12402, input_url="12401/train_loss")
])
