import numpy as np

from .aegis_node import AegisNode

#TODO: move to aegis_nodes package
class GaussianLerp(AegisNode):
  def __init__(self, port, mean=0, stddev=1, shape=None, steps=100):
    """Interpolates between points sampled from a normal distribution"""
    super().__init__(port)
    self.mean = mean
    self.stddev = stddev
    self.shape = shape
    self.step_counter = 0
    self.steps = steps
    self.a = self.sample()
    self.b = self.sample()

  def sample(self):
    return np.random.normal(loc=self.mean, scale=self.stddev, size=self.shape)

  def flip(self):
    self.a = self.b
    self.b = self.sample()

  def update(self):
    t = self.step_counter / self.steps
    x = self.a + (self.b - self.a) * t

    self.set_output(x)

    self.step_counter += 1
    if self.step_counter >= self.steps:
      self.step_counter = 0
      self.flip()
      print("flip")

#for testing purposes
if __name__ == "__main__":
  import threading
  import time
  from .print_node import PrintNode

  a = threading.Thread(
    target=lambda: GaussianLerp(12400, shape=[2], steps=100000).start(),
    daemon=True)
  a.start()

  b = threading.Thread(
    target=lambda: PrintNode(12401, input_url="12400", decimals=1).start(),
    daemon=True)
  b.start()

  #just to wait for ctrl c
  while(True):
    time.sleep(0)
