import numpy as np
import tensorflow as tf

from .engine import RequestEngine

from tensorflow.keras.applications import VGG16, VGG19, ResNet50, InceptionV3, \
  InceptionResNetV2, Xception, MobileNet, MobileNetV2, DenseNet121, DenseNet169, \
  DenseNet201, NASNetMobile, NASNetLarge
import tensorflow.keras.applications as apps

#TODO: not sure if this is needed...
#class, default input size, default output size, output depth, preprocessor
apps = {
  "vgg16": (VGG16, 224, 7, 512, apps.vgg16.preprocess_input),
  "vgg19": (VGG19, 224, 7, 512, apps.vgg19.preprocess_input),
  "resnet50": (ResNet50, 224, 7, 2048, apps.resnet50.preprocess_input),
  "inception_v3": (InceptionV3, 299, 8, 2048, apps.inception_v3.preprocess_input),
  "inception_resnet_v2": (InceptionResNetV2, 299, 8, 1536, apps.inception_resnet_v2.preprocess_input),
  "xception": (Xception, 299, 10, 2048, apps.xception.preprocess_input),
  "mobilenet": (MobileNet, 224, 7, 1024, apps.mobilenet.preprocess_input),
  "mobilenet_v2": (MobileNetV2, 224, 7, 1280, apps.mobilenet_v2.preprocess_input),
  "densenet121": (DenseNet121, 224, 7, 1024, apps.densenet.preprocess_input),
  "densenet169": (DenseNet169, 224, 7, 1664, apps.densenet.preprocess_input),
  "densenet201": (DenseNet201, 224, 7, 1920, apps.densenet.preprocess_input),
  "nasnet_mobile": (NASNetMobile, 224, 7, 1056, apps.nasnet.preprocess_input),
  "nasnet_large": (NASNetLarge, 331, 11, 4032, apps.nasnet.preprocess_input),
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

    app, in_size, out_size, out_depth, preprocessor = apps[app_name]
    self.model = app(include_top=False, weights="imagenet",
      input_shape=None, pooling=pooling)

    self.preprocess_input = preprocessor

    self.input_shape = [in_size, in_size, 3]
    self.output_shape = [out_depth]

  def update(self, reward):
    input_states = self.get_inputs(0) #dont propagate reward
    #fix Nones
    input_states = [np.zeros(self.input_shape) if x is None else x for x in input_states]
    #cast
    input_states = [x.astype("float32") for x in input_states]
    #preprocess inputs
    #TODO: inputs better be the same size ;)
    input_states = np.array([self.preprocess_input(x) for x in input_states])
    output_states = self.model(input_states)
    #TODO: mean? maybe allow returning the batch?
    output_state = np.mean(output_states, axis=0)

    return output_state
