import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import sys

from .aegis_node import AegisNode
from .utils import plt_pause

def ema(x, alpha=0.01):
  mean = []
  variance = []
  for val in x:
    val = 0 if math.isnan(val) or val is None else val #ree

    if len(mean) == 0:
      mean.append(val)
      variance.append(0)
    else:
      diff = val - mean[-1]
      mean.append(mean[-1] + diff * alpha)
      variance.append(variance[-1] + (abs(diff) - variance[-1]) * alpha)
  mean = np.array(mean)
  variance = np.array(variance)
  return (mean, variance)

import random

#TODO: implement smoothing
def sample(iterable, n):
    """
    Returns @param n random items from @param iterable.
    """
    reservoir = []
    for t, item in enumerate(iterable):
        if t < n:
            reservoir.append(item)
        else:
            m = random.randint(0,t)
            if m < n:
                reservoir[m] = item
    return reservoir

#who needs tensorboard anyway?
#TODO: ability to save/load data?
class GraphNode(AegisNode):
  def __init__(self, port, input_url, title=None, size=10000, smoothing=1):
    #use default input/output names
    super().__init__(port, inputs=input_url)
    self.title = title
    self.smoothing = smoothing
    self.graph = None
    self.res = []
    self.resx = []
    self.size = size
    self.count = 0
    plt.ion()

  def update(self):
    data = self.get_input()

    if data is not None:
      if self.count < self.size:
          self.res.append(data)
          self.resx.append(self.count)
      else:
          m = random.randint(0, self.count)
          if m < self.size:
              #reservoir[m] = item
              del self.res[m]
              del self.resx[m]
              self.res.append(data)
              self.resx.append(self.count)
      self.count += 1

      if self.graph is None:
        self.graph = plt.figure()
        plt.plot(self.resx, self.res)
        if self.title is not None:
          plt.title(self.title)
        plt.show()
      else:
        ax = self.graph.gca()
        ax.clear()
        ax.plot(self.resx, self.res)
        if self.title is not None:
          plt.title(self.title)
      plt_pause(0.0001) #??

if __name__ == '__main__':
  #TODO
  import argparse
  from aegis_core.utils import start_nodes

  parser = argparse.ArgumentParser()
  parser.add_argument("--port", "-p", type=int)
  parser.add_argument("--device", "-d", type=int, default=0)
  parser.add_argument("--fovea-url", "-f", type=str, default=None)
  parser.add_argument("--fovea-size", "-F", type=float, default=1./4)
  parser.add_argument("--resize", "-s", type=int, default=None) #TODO: support 2d resize
  parser.add_argument("--color", "-c", type=str, default="rgb")
  #TODO: add arg for color scale?

  parser.add_argument("--niceness", "-N", type=float, default=1.)
  parser.add_argument("--delay", "-D", type=float, default=1./100)

  args = parser.parse_args()

  node = CameraNode(
    port=args.port,
    device=args.device,
    fovea_url=args.fovea_url,
    fovea_size=args.fovea_size,
    resize=args.resize,
    color=args.color,
  ).set_niceness(args.niceness).set_delay(args.delay)

  start_nodes([node])
