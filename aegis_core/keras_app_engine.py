import numpy as np
import tensorflow as tf

from ml_utils.keras import get_states, set_states, apply_regularization
from ml_utils.viz import viz_weights

from pget.pget import create_traces, update_traces, step_weights
from pget.pget import explore_continuous, explore_discrete, categorical_crossentropy

from .engine import RequestEngine

from tensorflow.keras.applications import VGG16, VGG19, ResNet50, InceptionV3, \
  InceptionResNetV2, Xception, MobileNet, MobileNetV2, DenseNet121, DenseNet169, \
  DenseNet201, NASNetMobile, NASNetLarge

#TODO: not sure if this is needed...
#class, default input size, default output size, output depth
apps = {
  "vgg16": (VGG16, 224, 7, 512),
  "vgg19": (VGG19, 224, 7, 512),
  "resnet50": (ResNet50, 224, 7, 2048),
  "inception_v3": (InceptionV3, 299, 8, 2048),
  "inception_resnet_v2": (InceptionResNetV2, 299, 8, 1536),
  "xception": (Xception, 299, 10, 2048),
  "mobilenet": (MobileNet, 224, 7, 1024),
  "mobilenet_v2": (MobileNetV2, 224, 7, 1280),
  "densenet121": (DenseNet121, 224, 7, 1024),
  "densenet169": (DenseNet169, 224, 7, 1664),
  "densenet201": (DenseNet201, 224, 7, 1920),
  "nasnet_mobile": (NASNetMobile, 224, 7, 1056),
  "nasnet_large": (NASNetLarge, 331, 11, 4032),
  #TODO: included in tf 2.0
  #"resnet101": ResNet101,
  #"resnet152": ResNet152,
  #"resnet50_v2": ResNet50V2,
  #"resnet101_v2": ResNet101V2,
  #"resnet152_v2": ResNet152V2
}

#TODO: remove pooling requirement
#inputs should be rgb images 0-255 i believe

class KerasAppEngine(RequestEngine):
  def __init__(self, app_name="mobilenet_v2", pooling="max", input_urls=[]):
    super().__init__(input_urls=input_urls)

    app, in_size, out_size, out_depth = apps[app_name]
    self.model = app(include_top=False, weights="imagenet",
      input_shape=None, pooling=pooling)

    self.input_shape = [in_size, in_size, 3]
    self.output_shape = [out_depth]

  def update(self, reward):
    input_states = self.get_inputs(0) #dont propagate reward
    #fix Nones
    input_states = [np.zeros(self.input_shape) if x is None else x for x in input_states]
    #TODO: batch if all inputs are guaranteed to be the same dimensions?
    output_states = [self.model(input_state) for input_state in input_states]
    output_state = np.mean(output_states, axis=0)

    return output_state
