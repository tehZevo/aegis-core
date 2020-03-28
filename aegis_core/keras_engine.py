import numpy as np
import tensorflow as tf
import cv2

from .engine import RequestEngine

#TODO: move to separate repo

#assumes model has one input and one output
#TODO: remove requirement that io shapes are fully defined
class KerasEngine(RequestEngine):
  """Use a frozen Keras model as a node"""
  def __init__(self, model, input_urls=[]):
    super().__init__(input_urls=input_urls)

    self.model = model

    self.input_shape = self.model.input_shape[1:] #skip batch dim
    self.output_shape = self.model.output_shape[1:]

  def update(self, reward):
    input_states = self.get_inputs(0) #dont propagate reward
    #fix Nones
    input_states = [np.zeros(self.input_shape) if x is None else x for x in input_states]
    #cast
    input_states = [x.astype("float32") for x in input_states]
    input_states = np.sum(input_states, 0) #TODO: other methods for merging (or not merging at all)
    input_states = np.expand_dims(input_states, 0) #add batch dimension
    output_state = self.model(input_states)[0]

    return output_state
