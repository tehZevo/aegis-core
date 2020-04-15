import numpy as np
import tensorflow as tf

from .aegis_node import AegisNode

#TODO: move to separate repo maybe
#TODO: support multibinary/multidiscrete
#TODO: temperature

#kinda just an activation node for now lol
class DiscretizerNode(AegisNode):
  """Discretizes input"""
  def __init__(self, inputs=[], niceness=1, cors=True):
    super().__init__(inputs=inputs, niceness=niceness, cors=cors)

  def update(self):
    input_states = self.get_input()

    #TODO: checks/warnings for missing inputs? or will AegisNode warn us about that?
    input_states = [x.astype("float32") for x in input_states if x is not None]
    if len(input_states) == 0:
      input_states = None
    else:
      input_states = np.sum(input_states, 0) #TODO: other methods for merging (or not merging at all)
    output_state = tf.nn.softmax(input_states).numpy()

    return output_state

if __name__ == "__main__":
  import logging
  log = logging.getLogger('werkzeug')
  log.setLevel(logging.ERROR)

  tf.enable_eager_execution()

  DiscretizerNode(12400, []).start() #TODO
