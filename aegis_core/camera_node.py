import numpy as np
import tensorflow as tf
import cv2
import numbers

from .aegis_node import AegisNode

class CameraNode(AegisNode):
  def __init__(self, port, device=0, fovea_url=None, fovea_size=1./4, resize=None, color="rgb", scale=1/255.):
    #use default input/output names
    self.cap = cv2.VideoCapture(device)
    self.resize = resize
    self.color = color
    self.scale = scale

    self.use_fovea = fovea_url is not None
    inputs = None
    if self.use_fovea:
      if isinstance(fovea_size, numbers.Number):
        fovea_size = (fovea_size, fovea_size)
      if fovea_size[0] > 1 or fovea_size[0] < 0 or fovea_size[1] > 1 or fovea_size[1] < 0:
        raise ValueError("Fovea size must be between 0 and 1 on both axes")

      inputs = {"fovea": fovea_url}

    self.fovea_size = fovea_size

    super().__init__(port, inputs=inputs)

  def update(self):
    #TODO: scale by 1/255 (self.scale)
    ret, img = self.cap.read()

    bounds = [0, 0, 1, 1] #xywh
    w = img.shape[0] #TODO: check if these need to be swapped
    h = img.shape[1]

    if self.use_fovea:
      x, y = 0.5, 0.5
      xy = self.get_input("fovea")
      if xy is not None:
        x, y = xy

      left = x - (x * self.fovea_size[0])
      top = y - (y * self.fovea_size[1])
      bounds = [left, top, left + self.fovea_size[0], top + self.fovea_size[1]]

    bounds = np.array(bounds) * np.array([w, h, w, h])
    bounds = [int(e) for e in bounds]

    #clip image according to bounds:
    img = img[bounds[0]:bounds[2], bounds[1]:bounds[3]]

    if self.resize is not None:
      if isinstance(self.resize, tuple):
        w, h = self.resize
      else: #better be a number-like
        w, h = (self.resize, self.resize)

      img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR) #TODO: option for using nearest neighbor?

    if self.color == "gray":
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif self.color == "rgb":
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #else just bgr

    img = img.astype("float32")
    img = img * self.scale
    self.set_output(img)

def camera_node(port, device=0, fovea_url=None, fovea_size=1./4, resize=None, color="rgb", niceness=1., delay=1./100):
  args = ["python", "-m", "aegis_core.camera_node"]
  args += ["-d", device]
  args += ["-f", fovea_url] if fovea_url is not None else []
  args += ["-F", fovea_size]
  args += ["-r", resize] if resize is not None else []
  args += ["-c", color]
  args += ["-N", niceness]
  args += ["-D", delay]

  return " ".join(args)

if __name__ == '__main__':
  import argparse
  from aegis_core.utils import start_nodes

  parser = argparse.ArgumentParser()
  parser.add_argument("--port", "-p", type=int)
  parser.add_argument("--device", "-d", type=int, default=0)
  parser.add_argument("--fovea-url", "-f", type=str, default=None)
  parser.add_argument("--fovea-size", "-F", type=float, default=1./4)
  parser.add_argument("--resize", "-r", type=int, default=None) #TODO: support 2d resize
  parser.add_argument("--color", "-c", type=str, default="rgb")
  parser.add_argument("--color-scale", "-s", type=float, default=1./255)

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
