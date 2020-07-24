from collections import deque
import random

import numpy as np
import tensorflow as tf

from .aegis_node import AegisNode

from keras.layers import Dense, Reshape, Flatten
from keras.models import Sequential
from dehydrated_vae import build_vae

#TODO: modify LocalCuriosityEngine logic to fit VAE
#local curiosity engine:
# has a model of its own (autoencoder of some sort), trains every step on input
# node, sends loss as reward to action node or proxy
# maybe add noise so the autoencoder has to denoise as well? idk...
# what types of noise to use for denoising AE?

# functions of vae node:
# training vae
# compressing inputs
# curiosity
# generating new samples

#TODO: saving callback?
#TODO: callback for visualizing latent space?
#TODO: internal buffer + training steps.. optimizer etc
# need to update mlutils model builder to optionally include optimizer
#TODO: need to have separate loss because vae loss will include kl...
class VAENode(AegisNode):
  def __init__(self, port, model, encoder, decoder, input_url=None, generator_url=None, train=True,
      buffer_size=10000, batch_size=32, loss=tf.keras.losses.mean_squared_error):
    """Model should be compiled first"""
    #TODO: support multiple encoder/generator urls
    # create one channel for each encoder/generator input
    # for losses, loop over each input and sum/average loss
    inputs = {}
    if input_url is not None:
      inputs["encoder"] = input_url
    if generator_url is not None:
      inputs["generator"] = generator_url

    outputs = []
    if input_url is not None:
      outputs.append("encoder")
      outputs.append("test_loss") #TODO: rename (baseline_loss?)
      outputs.append("surprise")
      outputs.append("reconstruction")
    if generator_url is not None:
      outputs.append("generator")
    if train:
      outputs.append("train_loss")

    super().__init__(port, inputs=inputs, outputs=outputs)
    #self.niceness = 0  #TODO: REMOVE ME
    self.model = model
    self.encoder = encoder
    self.decoder = decoder
    self.train = train
    self.buffer = deque(maxlen=buffer_size)
    self.batch_size = batch_size
    self.loss = loss

  def get_batch(self):
    batch = random.choices(self.buffer, k=self.batch_size)
    batch = np.array(batch)
    return batch

  def encode(self, x):
    #TODO: might have to [0] if encoder has 2 outputs
    return self.encoder.predict_on_batch(x)

  def decode(self, x):
    return self.decoder.predict_on_batch(x)

  def calc_loss(self, batch):
    #TODO: if inputs are different shapes, loop and call test_on_batch separately
    x = batch
    pred_latent = self.encode(x)
    pred_x = self.decode(pred_latent)
    loss = np.mean(self.loss(x, pred_x)) #lol mse acts on last axis
    loss = float(loss)
    return loss

  def update(self):
    #encode and calculate test loss/surprise (TODO: only supports 1 input currently)
    encoder_input = self.get_input("encoder") if self.has_input("encoder") else None

    if encoder_input is not None:
      #technically, we should wait until after we calculate surprise, but this
      # is easier, and has the nice effect that the initial surprise should be 0
      #add input to buffer
      self.buffer.append(encoder_input) #buffer should trim old elements automatically
      encoder_input = np.expand_dims(encoder_input, 0)

      #calculate surprise (use batch)
      baseline_loss = self.calc_loss(self.get_batch())
      encoding = self.encode(encoder_input)
      reconstruction = self.decode(encoding)
      sample_loss = float(np.mean(self.loss(encoder_input, reconstruction))) #lol mse acts on last axis

      #use previous training loss as a baseline
      surprise = sample_loss - baseline_loss
      #set loss/surprise
      self.set_output(sample_loss, "test_loss")
      #print("ree", sample_loss, len(self.buffer))
      self.set_output(surprise, "surprise")

      #trim off batch dimensions
      self.set_output(encoding[0], "encoder")
      self.set_output(reconstruction[0], "reconstruction")

    #train
    if self.train and len(self.buffer) > 0:
      x = self.get_batch()
      #TODO: assumes no metrics (returns loss only)
      #train_loss = self.model.train_on_batch(x, x)
      train_loss = float(self.model.train_on_batch(x, x, return_dict=True))
      #print("hi", train_loss, len(self.buffer))
      #train loss will be whatever the model defines (including kl)
      self.set_output(train_loss, "train_loss")

    #generation
    generator_input = self.get_input("generator") if self.has_input("generator") else None
    if generator_input is not None:
      x = np.expand_dims(generator_input, 0)
      generation = self.decoder.predict(x)[0]
      self.set_output(generation, "generator")
