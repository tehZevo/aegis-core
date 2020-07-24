from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, MaxPooling2D, UpSampling2D

import time
import numpy as np

from aegis_core.keras_dataset_node import KerasDataset
from aegis_core.imshow_node import ImshowNode
from aegis_core.random_node import RandomNode
from aegis_core.gaussian_lerp import GaussianLerp
from aegis_core.vae import VAENode
from aegis_core.print_node import PrintNode
from aegis_core.graph_node import GraphNode
from aegis_core.utils import start_nodes, build_vae

#TODO
#immediate goal:
#train vae using camera node
#  visualize source image, reconstruction, and latent generation
#  log vae loss, test loss, and surprise/sample loss

#TODO: create input source (mnist/camera node)
#TODO: log training loss (print node)
#TODO: log encoder latent (print node)
#TODO: later, test generation
#TODO: cv/matplotlib based image visualizer node :)
#TODO: use multiprocessing for utils.start_node so we dont have to deal with threads?


#NOTE: success!
# i held my hand in front of the camera for many frames, so the VAE learned 2 images: with hand and without hand
# then, when i removed my hand, and moved my head into the area where my hand was, it began drawing my hand over my face
# of course, putting my hand up there also adds my hand into the reconstruction as well
if __name__ == '__main__':
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

  # encoder = Sequential([
  #   Conv2D(16, 3, padding="same", activation=acti, input_shape=input_shape),
  #   MaxPooling2D(),
  #   Conv2D(32, 3, padding="same", activation=acti),
  #   MaxPooling2D(),
  #   Conv2D(64, 3, padding="same", activation=acti),
  #   MaxPooling2D(),
  #   Flatten(),
  #   Dense(64, activation=acti),
  # ])
  #
  # inner_size = int(input_shape[0] / 2 / 2 / 2) #assumes square
  #
  # decoder = Sequential([
  #   Dense(inner_size * inner_size, input_shape=[latent_size], activation=acti),
  #   Reshape([inner_size, inner_size, 1]),
  #   Conv2D(64, 3, padding="same", activation=acti),
  #   UpSampling2D(),
  #   Conv2D(32, 3, padding="same", activation=acti),
  #   UpSampling2D(),
  #   Conv2D(16, 3, padding="same", activation=acti),
  #   UpSampling2D(),
  #   Conv2D(3, 3, padding="same", activation="sigmoid"),
  # ])

  vae, encoder, decoder = build_vae(encoder, decoder, input_shape, latent_size)
  vae.summary()
  encoder.summary()
  decoder.summary()

  start_nodes([
    #RandomNode(12399, shape=[2]),
    #CameraNode(12400, resize=32),
    #CameraNode(12400, fovea_url="12399", fovea_size=1./4, resize=32),
    KerasDataset(12400, dataset="mnist", steps=10),
    ImshowNode(12401, input_url="12400", title="source"),
    GaussianLerp(12402, shape=[latent_size], steps=10),
    VAENode(12403, vae, encoder, decoder, input_url="12400", generator_url="12402"),
    PrintNode(12404, input_url="12403/train_loss", prefix="train loss"),
    ImshowNode(12405, input_url="12403/reconstruction", title="reconstruction"),
    ImshowNode(12406, input_url="12403/generator", title="generation"),
    GraphNode(12407, input_url="12403/train_loss", size=1000, title="train_loss")
  ])
