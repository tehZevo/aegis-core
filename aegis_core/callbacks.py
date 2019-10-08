import tensorflow as tf

#WIP
class AegisCallback():
  def __init__(self, interval):
    self.interval = interval
    self.step_counter = 0

  def do_callback(self, data):
    pass

  def __call__(self, data):
    self.step_counter += 1
    if self.step_counter >= self.interval:
      self.step_counter = 0
      self.do_callback(data)

#logs
# node_run_name
#

#TODO: how to handle step vs episode...?
#TODO: accumulate
class TensorboardCallback(AegisCallback):
  """ Requires TF eager to be enabled """
  def __init__(self, writer, field, summary_type="scalar", prefix=""):
    super().__init__(interval=1)
    self.writer = writer
    self.step = 0
    self.prefix = prefix
    self.field = field

    #TODO: support other types
    s = tf.contrib.summary
    stype = summary_type.lower()
    self.summary_type = (s.scalar if stype == "scalar"
      else s.histogram if stype == "histogram"
      else s.text)

  def do_callback(self, data):
    self.summary_type(self.prefix + "/" + self.field, data[self.field], self.step)
    self.step += 1

  def __call__(self, data):
    with self.writer.as_default(), tf.contrib.summary.always_record_summaries():
      super().__call__(data)

#TODO: get path from engine
class WeightVisualizer(AegisCallback):
  def __init__(self, path, interval=1000):
    super().__init__(interval)
    self.path = path

  def do_callback(self, engine):
    print("plotting weights to {}".format(self.path))
    viz_weights(engine.model.get_weights(), self.path + ".png")

#TODO: get path from engine
class ModelSaver(AegisCallback):
  def __init__(self, path, interval=1000):
    super().__init__(interval)
    self.path = path

  def __call__(self):
    self.steps_since_save += 1
    if self.steps_since_save > self.save_interval:
      print("saving model to {}".format(self.path))
      self.model.save(self.path)
      self.steps_since_save = 0

class RewardPrinter(AegisCallback):
  def __init__(self, interval=100):
    super().__init__(interval)
    self.rewards = []
