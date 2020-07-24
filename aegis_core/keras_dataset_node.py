import numpy as np
from keras.datasets import mnist, cifar10, cifar100
import cv2

from .aegis_node import AegisNode

#TODO: move to aegis_nodes package
class KerasDataset(AegisNode):
  """Outputs random mnist/cifar samples
  dataset options are mnist, cifar10, and cifar100
  """
  def __init__(self, port, steps=100, dataset="mnist", test=False, resize=None):
    """Interpolates between points sampled from a normal distribution"""
    super().__init__(port)
    self.step_counter = 0
    self.steps = steps
    self.dataset_name = dataset.strip().lower()
    self.resize = resize

    dataset = self.dataset_name
    if dataset not in ["mnist", "cifar", "cifar10", "cifar100"]:
      raise ValueError("dataset should be mnist, cifar10, or cifar100")
    dataset = cifar10 if dataset == "cifar10" or dataset == "cifar" else cifar100 if dataset == "cifar100" else mnist

    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    #TODO: allow for channels_first

    dataset = x_test if test else x_train
    dataset = dataset.astype('float32')
    dataset /= 255.
    #add dim to mnist to make it in-line with cifar
    if self.dataset_name == "mnist":
      dataset = np.expand_dims(dataset, -1)

    print('dataset shape:', dataset.shape)
    print(dataset.shape[0], 'dataset samples')

    self.dataset = dataset

    self.flip()

  def flip(self):
    i = np.random.randint(self.dataset.shape[0])
    x = self.dataset[i]

    if self.resize is not None:
      if isinstance(self.resize, tuple):
        w, h = self.resize
      else: #better be a number-like
        w, h = (self.resize, self.resize)

      x = cv2.resize(x, (w, h)) #TODO: option for using nearest neighbor?

    #resize removes last dim if it's 1....
    if len(x.shape) == 2:
      x = np.expand_dims(x, -1)

    self.sample = x

  def update(self):

    self.set_output(self.sample)

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
    target=lambda: KerasDataset(12400, steps=100000).start(),
    daemon=True)
  a.start()
  #
  # b = threading.Thread(
  #   target=lambda: PrintNode(12401, input_url="12400", decimals=1).start(),
  #   daemon=True)
  # b.start()

  #just to wait for ctrl c
  while(True):
    time.sleep(0)
