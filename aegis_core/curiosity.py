from collections import deque
import random

import numpy as np
import tensorflow as tf

from ml_utils.viz import viz_weights
from .engine import RequestEngine, Engine, request_input, sanitize

class CuriosityEngine(Engine):
  """
  Takes pred/true input nodes, calculates loss, sends loss as reward to
  action node (or proxy), sends -loss as reward to pred node (or proxy)
  """
  def __init__(self, true_url, pred_url, action_url, pred_proxy=None,
      loss=tf.keras.losses.mean_squared_error):
    super().__init__()
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
#TODO: internal buffer + training steps.. optimizer etc
# need to update mlutils model builder to optionally include optimizer
class LocalCuriosityEngine(RequestEngine):
  def __init__(self, model, action_url, input_urls=[], train=True,
      buffer_size=10000, batch_size=32, callbacks=[]):
    """Model should be compiled first"""
    super().__init__(input_urls=input_urls)
    self.model = model
    self.action_url = sanitize(action_url)
    self.train = train
    self.buffer = deque()
    self.buffer_size = buffer_size
    self.batch_size = batch_size
    self.callbacks = callbacks

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
    if train:
      batch = random.sample(self.buffer, self.batch_size)
      x = np.array(batch)
      train_loss = self.model.train_on_batch(x, x)

    cb_values = {}
    cb["surprise"] = surprise
    cb["loss"] = train_loss
    cb["engine"] = self

    for cb in self.callbacks:
      cb(cb_values)

    #TODO: return something more useful? idk
    return np.mean(inputs, axis=0)
