import numpy as np

from .aegis_node import AegisNode

#TODO: move to aegis_nodes package
class RandomNode(AegisNode):
  def __init__(self, port, min=0, max=1, shape=None):
    """Interpolates between points sampled from a normal distribution"""
    super().__init__(port)
    self.min = min
    self.max = max
    self.shape = shape

  def update(self):
    self.set_output(np.random.uniform(low=self.min, high=self.max, size=self.shape))
