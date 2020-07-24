import numpy as np
import tensorflow as tf

from .aegis_node import AegisNode

#TODO: move to separate repo

#assumes model has one input and one output
#TODO: remove requirement that io shapes are fully defined
#TODO: support models with multiple ins/outs
class KerasNode(AegisNode):
  """Use a Keras model as an inference node
  provide input_shape to replace Nones with np.zeros for partially-defined
  input shapes
  """
  def __init__(self, model, input_url, input_shape=None):
    #use default input/output names
    super().__init__(inputs=input_url)

    self.model = model
    #skip batch dim
    self.input_shape = self.model.input_shape[1:] if input_shape is None else input_shape

  def update(self, reward):
    input_state = self.get_input()

    if x is None and expected_input_shape is not None:
      #fix Nones
      input_state = np.zeros(self.expected_input_shape)

    #housekeeping
    input_state = x.astype("float32")
    input_state = np.expand_dims(input_state, 0) #add batch dimension
    output_state = self.model(input_state)[0].numpy() #TODO: assumes eager i think..
    #i believe this was to avoid some memory leak when repeatedly calling predict

    self.set_output(output_state)
