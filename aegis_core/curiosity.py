from collections import deque
import random

import numpy as np
import tensorflow as tf

#TODO: convert to using aegisnode
from .aegis_node import AegisNode

class CuriosityEngine(AegisNode):
  """
  Takes pred/true input nodes, calculates loss, sends loss as reward to
  action node (or proxy), sends -loss as reward to pred node (or proxy)
  """
  def __init__(self, port, true_url, pred_url, action_url, pred_proxy=None,
      loss=tf.keras.losses.mean_squared_error, callbacks=None):
    inputs = {
      "true": true_url,
      "pred": pred_url,
      "action": action_url,
      "reward": pred_proxy if pred_proxy is not None else pred_url
    }
    super().__init__(port, inputs=inputs, callbacks=callbacks)
    self.true_url = sanitize(true_url)
    self.pred_url = sanitize(pred_url)
    self.action_url = sanitize(action_url)
    self.pred_proxy = None if pred_proxy is None else sanitize(pred_proxy)
    self.loss_func = loss

  def update(self, reward):
    #get inputs and calculate loss
    true = request_input(self.true_url)
    pred = request_input(self.pred_url)
    loss = self.loss_func(true, pred)

    #reward action
    request_input(action_url, loss)

    #punish (train) predictor
    pred_url = self.pred_proxy if self.pred_proxy is not None else self.pred_url
    request_input(pred_url, -loss)

    return pred


#local curiosity engine:
# has a model of its own (autoencoder of some sort), trains every step on input
# node, sends loss as reward to action node or proxy
# maybe add noise so the autoencoder has to denoise as well? idk...
# what types of noise to use for denoising AE?

#TODO: saving callback?
#TODO: callback for visualizing latent space?
#TODO: internal buffer + training steps.. optimizer etc
# need to update mlutils model builder to optionally include optimizer
class LocalCuriosityEngine(RequestEngine):
  def __init__(self, model, action_url, input_urls=[], train=True,
      buffer_size=10000, batch_size=32, callbacks=[], subtract_train_loss=False):
    """Model should be compiled first"""
    super().__init__(input_urls=input_urls)
    self.model = model
    self.action_url = sanitize(action_url)
    self.train = train
    self.buffer = deque()
    self.buffer_size = buffer_size
    self.batch_size = batch_size
    self.callbacks = callbacks
    self.subtract_train_loss = subtract_train_loss
    self.last_train_loss = None

    #extract io shapes from model
    self.input_shape = self.model.input_shape[1:]
    self.output_shape = self.model.output_shape[1:]

  def update(self, reward):
    #get inputs
    inputs = self.get_inputs(0)
    #fix nones
    inputs = [np.zeros(self.input_shape) if x is None else x for x in inputs]
    #calculate loss (surprise)
    #TODO: assumes the model has no metrics, only loss
    x = np.array(inputs)
    surprise = self.model.test_on_batch(x, x)
    surprise = float(surprise)
    #use previous training loss as a baseline of sorts
    if self.subtract_train_loss:
      surprise = (surprise - self.last_train_loss
        if self.last_train_loss is not None else 0)
    #reward action node/proxy with surprise
    request_input(self.action_url, surprise)

    #add inputs to batch TODO: make it so it doesnt add zeros from Nones?
    for s in inputs:
      self.buffer.append(s)
    #trim buffer
    while len(self.buffer) > self.buffer_size:
      self.buffer.popleft()

    train_loss = 0

    #train
    if self.train:
      batch = random.choices(self.buffer, k=self.batch_size)
      x = np.array(batch)
      train_loss = self.model.train_on_batch(x, x)

    self.last_train_loss = train_loss

    cb_values = {}
    cb_values["surprise"] = surprise
    cb_values["loss"] = train_loss
    cb_values["engine"] = self

    for cb in self.callbacks:
      cb(cb_values)

    #TODO: return something more useful? idk
    return np.mean(inputs, axis=0)
