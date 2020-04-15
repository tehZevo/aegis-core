import numpy as np

from .aegis_node import AegisNode

#TODO: move to aegis_nodes package
class GaussianLerp(AegisNode):
  def __init__(self, port, mean=0, stddev=1, shape=None, steps=100, cors=True, niceness=1):
    """Interpolates between points sampled from a normal distribution"""
    super().__init__(port, niceness=niceness, cors=cors)
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

    self.set_state(x)

    self.step_counter += 1
    if self.step_counter >= self.steps:
      self.step_counter = 0
      self.flip()
      print("flip")

#for testing purposes
if __name__ == "__main__":
  GaussianLerp(12400, shape=[10], steps=100000).start()
