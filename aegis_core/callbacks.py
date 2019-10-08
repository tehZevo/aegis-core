import tensorflow as tf
import numpy as np

#WIP
class AegisCallback():
  def __init__(self, interval):
    self.interval = interval
    self.step_counter = 0
    self.call_counter = 0

  def do_callback(self, data):
    pass

  def __call__(self, data):
    self.step_counter += 1
    if self.interval is None or self.step_counter >= self.interval:
      self.step_counter = 0
      self.call_counter += 1
      self.do_callback(data)

class ValueCallback(AegisCallback):
  def __init__(self, field, interval=None, reduce="sum"):
    """Reduce can be either "mean", "sum", or None, in which case, all values
    are used. If a string interval is provided, the callback will wait
    until data[interval] is truthy.
    """
    super().__init__(interval=interval)
    self.field = field
    self.reduce_method = reduce
    self.values = []

  def do_callback(self, value):
    pass

  def __call__(self, data):
    self.values.append(data[self.field])
    self.step_counter += 1
    interval_is_str = type(self.interval) is str
    if ((interval_is_str and data[self.interval]) or
        (not interval_is_str and self.step_counter >= self.interval)):
      self.step_counter = 0
      self.call_counter += 1
      #reduce
      self.value = (np.mean(self.values, axis=0) if self.reduce_method == "mean" else
        np.sum(self.values, axis=0) if self.reduce_method == "sum" else self.values)
      self.values = []
      self.do_callback(self.value)

#TODO
class GraphCallback(ValueCallback):
  def __init__(self, path, field, interval=None, title=None):
    pass

class TensorboardCallback(ValueCallback):
  """ Requires TF eager to be enabled """
  def __init__(self, writer, field, interval=None, suffix="",
       summary_type="scalar", reduce="sum", step_for_step=True):
    super().__init__(field, interval=interval, reduce=reduce)
    self.writer = writer
    self.step = 0
    self.suffix = suffix
    self.step_for_step = step_for_step

    #TODO: support other types
    s = tf.contrib.summary
    stype = summary_type.lower()
    self.summary_type = (s.scalar if stype == "scalar"
      else s.histogram if stype == "histogram"
      else s.text)

  def do_callback(self, value):
    self.summary_type(self.field + "/" + self.suffix, value, step=self.step)

    if not self.step_for_step:
      self.step += 1

  def __call__(self, data):
    with self.writer.as_default(), tf.contrib.summary.always_record_summaries():
      if self.step_for_step:
        self.step += 1
      super().__call__(data)

class ValuePrinter(ValueCallback):
  def __init__(self, field, interval=None, reduce="sum", interval_name="Ep"):
    super().__init__(interval=interval, reduce=reduce)
    interval_name = interval_name

  def do_callback(self, value):
    print("{} {}: {}".format(self.interval_name, self.call_counter, value))

#TODO: get path from engine
class WeightVisualizer(AegisCallback):
  def __init__(self, path, interval=None):
    super().__init__(interval)
    self.path = path

  def do_callback(self, engine):
    print("plotting weights to {}".format(self.path))
    viz_weights(engine.model.get_weights(), self.path + ".png")

#TODO: get path from engine
class ModelSaver(AegisCallback):
  def __init__(self, path, interval=None):
    super().__init__(interval)
    self.path = path

  def __call__(self):
    self.steps_since_save += 1
    if self.steps_since_save > self.save_interval:
      print("saving model to {}".format(self.path))
      self.model.save(self.path)
      self.steps_since_save = 0
