import threading
import tensorflow as tf
from multiprocessing import Process
import asyncio

import matplotlib.pyplot as plt
import matplotlib
import sys

from keras.models import Model
import keras
from keras.layers import Dense
import numpy as np

def nullable_string(val):
  return val if val else None

def plt_pause(interval):
    manager = plt._pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw_idle()
        #plt.show(block=False)
        canvas.start_event_loop(interval)
    else:
        time.sleep(interval)

def start_node(node, niceness=1):
  print("HI")
  node.niceness = niceness
  return node, node.start()

def start_nodes(nodes):
  async def main():
    cs = [node.start() for node in nodes]
    await asyncio.gather(*cs)
  asyncio.run(main())

def start_node_thread(node, niceness=1):
  #with tf.Graph().as_default() as g:
  node.niceness = niceness
  thread = threading.Thread(target=lambda: node.start(), daemon=True)
  thread.start()
  return node, thread

def start_node_mp(node, niceness=1):
  def f():
    node.niceness = niceness
    node.start()
  p = Process(target=f)
  p.start()

  return node, p

##TODO: move VAE stuff to another package (integrate into dehydratedvae?)
def build_encoder(pre_encoder, latent_dim):
  encoder = pre_encoder
  encoder_inputs = encoder.inputs[0]
  x = encoder_inputs
  x = encoder(x)
  z_mean = Dense(latent_dim, name="z_mean")(x)
  z_log_var = Dense(latent_dim, name="z_log_var")(x)
  z = Sampling()([z_mean, z_log_var])
  encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
  #encoder2 = Model(encoder_inputs, z)
  encoder2 = Model(encoder_inputs, z_mean) #return z_mean in encoder so we arent looking through random samples
  return encoder, encoder2

class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, loss=tf.keras.losses.mean_squared_error, kl_scale=1., **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_scale = kl_scale
        self.lossfunc = loss

    def call(self, inputs):
      z_mean, z_log_var, z = self.encoder(inputs)
      reconstruction = self.decoder(z)
      return reconstruction

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                self.lossfunc(data, reconstruction)
            )
            #reconstruction_loss *= 28 * 28 #TODO: assumed input shape
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            kl_loss /= np.prod(self.encoder.input_shape[1:]) #scale by 1/prod(inputshape)
            total_loss = reconstruction_loss + kl_loss * self.kl_scale
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # return {
        #     "loss": total_loss,
        #     "reconstruction_loss": reconstruction_loss,
        #     "kl_loss": kl_loss,
        # }
        return total_loss

def build_vae(pre_encoder, decoder, input_shape, latent_size, kl_scale=1., opt="adam"):
    encoder = pre_encoder
    encoder, encoder2 = build_encoder(encoder, latent_size)
    vae = VAE(encoder, decoder, kl_scale=kl_scale)

    #TODO: oh boy run eagerly...
    vae.compile(optimizer=opt, run_eagerly=True)

    vae.build((None,) + tuple(input_shape))

    return vae, encoder2, decoder
