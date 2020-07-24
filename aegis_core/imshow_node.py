import numpy as np
import tensorflow as tf

from .aegis_node import AegisNode
from .utils import plt_pause
import matplotlib.pyplot as plt
import matplotlib
import sys

import time

#TODO: 3d "images"
# https://matplotlib.org/3.1.1/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html
# https://matplotlib.org/3.1.1/_images/voxels_numpy_logo.png
# https://stackoverflow.com/questions/45729092/make-interactive-matplotlib-window-not-pop-to-front-on-each-update-windows-7/45734500
#TODO: make more efficient...
class ImshowNode(AegisNode):
  def __init__(self, port, input_url, cmap="gray", title=None):
    super().__init__(port, inputs=input_url)
    self.im = None
    self.cmap = cmap
    self.title = title
    plt.ion()

  def update(self):
    data = self.get_input()
    if data is not None:
      data = np.squeeze(data) #remove last [1] dim if exists

      if self.im is None:
        #TODO: check that this doesnt break for colored images
        plt.figure()
        self.im = plt.imshow(data, cmap=self.cmap)
        if self.title is not None:
          plt.title(self.title)
        plt.show()
        def handle_close(evt):
          #TODO: get this working
          print('Closed Figure!')
          sys.exit()
        plt.gcf().canvas.mpl_connect("close_event", handle_close)
      else:
        self.im.set_data(data)
      plt_pause(0.0001) #??
      #print(np.mean(data))

if __name__ == '__main__':
  import time

  from .keras_dataset_node import KerasDataset
  from .utils import start_node

  start_node(KerasDataset(12400, dataset="mnist", steps=100))
  #TODO: why is training loss 0 issue with loss function in dehydrated_vae?
  start_node(ImshowNode(12401, input_url="12400"))

  #just to wait for ctrl c
  while True:
    time.sleep(0)
