import numpy as np
import tensorflow as tf

from .engine import RequestEngine

#TODO: move to separate repo maybe
#TODO: support multibinary/multidiscrete
#TODO: temperature

#kinda just an activation node for now lol
class DiscretizerEngine(RequestEngine):
  """Discretizes input"""
  def __init__(self, size, input_urls=[]):
    super().__init__(input_urls=input_urls)

    self.size = size

    self.input_shape = self.size
    self.output_shape = self.size

  def update(self, reward):
    input_states = self.get_inputs(0) #no reward
    #fix Nones
    input_states = [np.zeros(self.input_shape) if x is None else x for x in input_states]
    #cast
    input_states = [x.astype("float32") for x in input_states]
    input_states = np.sum(input_states, 0) #TODO: other methods for merging (or not merging at all)
    input_states = np.expand_dims(input_states, 0) #add batch dimension
    output_states = self.model(input_states)[0]

    return output_state

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("-u", "--urls", nargs="+", required=True)
  parser.add_argument("-s", "--size", type=int, required=True)
  parser.add_argument('-p','--port', type=str, required=True)
  parser.add_argument('-n','--niceness', type=float, default=1)

  args = parser.parse_args()

  import logging
  log = logging.getLogger('werkzeug')
  log.setLevel(logging.ERROR)

  from aegis_core.flask_controller import FlaskController

  tf.enable_eager_execution()

  engine = DiscretizerEngine(args.size, args.urls)
  controller = FlaskController(engine, port=args.port, niceness=args.niceness)
