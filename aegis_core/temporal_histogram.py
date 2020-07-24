import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import sys
import seaborn as sns

from .aegis_node import AegisNode
from .utils import plt_pause

def reject_outliers(data, z=2):
    return data[abs(data - np.mean(data)) < z * np.std(data)]

class TemporalHistogram(AegisNode):
  def __init__(self, port, input_url, title=None, size=1000, outlier_z=2.):
    #use default input/output names
    super().__init__(port, inputs=input_url)
    self.title = title
    self.graph = None
    self.data = []
    self.size = size
    self.outlier_z = outlier_z
    plt.ion()

  def update(self):
    #assumes data is 1d..
    data = self.get_input()

    if data is not None:
      self.data.append(data)

      while len(self.data) > self.size:
        self.data.pop(0)

      if self.graph is None:
        self.graph = plt.figure()
        if self.title is not None:
          plt.title(self.title)
        plt.show()
      else:
        base_ax = self.graph.gca()
        self.graph.clear()
        d = np.array(self.data).T
        n = len(d)
        for i, dd in enumerate(d):
          if self.outlier_z is not None:
            dd = reject_outliers(dd, self.outlier_z)
          ax = plt.subplot(n, 1, i + 1)
          sns.distplot(dd)
          ax.axvline(data[i])
        if self.title is not None:
          base_ax.set_title(self.title)
      plt_pause(0.0001) #??
