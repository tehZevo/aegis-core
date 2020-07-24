import numpy as np
import tensorflow as tf

from .aegis_node import AegisNode

class PrintNode(AegisNode):
  def __init__(self, port, input_url, decimals=None, prefix=None):
    #use default input/output names
    super().__init__(port, inputs=input_url)
    self.prefix = prefix
    self.decimals = decimals

  def update(self):
    data = self.get_input()

    if self.decimals is not None and data is not None:
      data = np.round(data, self.decimals)

    if self.prefix is not None:
      print(self.prefix, data)
    else:
      print(data)
