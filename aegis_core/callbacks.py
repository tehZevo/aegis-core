import tensorflow as tf
import numpy as np

from ml_utils.viz import save_plot, viz_weights

class AegisCallback():
  def __init__(self, interval):
    """If a string interval is provided, the callback will wait until
    data[interval] is truthy.
    """
    self.interval = interval
    self.steps_since_call = 0
    self.step_counter = 0
    self.call_counter = 0

  def do_callback(self, data):
    pass

  def __call__(self, data):
    self.step_counter += 1
    self.steps_since_call += 1
    interval_is_str = type(self.interval) is str
    if ((interval_is_str and data[self.interval]) or
        (not interval_is_str and self.steps_since_call >= self.interval)):
      self.steps_since_call = 0
      self.call_counter += 1
      self.do_callback(data)

class LambdaCallback(AegisCallback):
  def __init__(self, interval, lam):
    super().__init__(interval)
    self.lam = lam

class ValueCallback(AegisCallback):
  def __init__(self, interval=None, reduce="sum"):
    """Reduce can be either "mean", "sum", "last"
    mean/sum: performs respective reduce op.
    last: returns only the last value.
    None: performs no reduce op, returns all values.
    """
    super().__init__(interval=interval)
    self.reduce_method = reduce
    self.values = []

  def get_value(self, data):
    pass

  def do_value_callback(self, value):
    pass

  def do_callback(self, data):
    #reduce
    if self.reduce_method == "mean":
      value = np.mean(self.values, axis=0)
    elif self.reduce_method == "sum":
      value = np.sum(self.values, axis=0)
    #only call get_value on last
    elif self.reduce_method == "last":
      value = self.get_value(data)
    else:
      value = self.values

    self.values = []

    self.do_value_callback(value)

  def __call__(self, data):
    if self.reduce_method != "last":
      self.values.append(self.get_value(data))
    super().__call__(data)

class FieldCallback(ValueCallback):
  def __init__(self, field, interval=None, reduce="sum"):
    super().__init__(interval=interval, reduce=reduce)
    self.field = field

  def get_value(self, data):
    return data[self.field]

#TODO: max length?
class GraphCallback(FieldCallback):
  """Saves accumulating matplotlib graphs of stuff"""
  def __init__(self, field, interval=None, title=None, reduce="mean",
      smoothing=0.1, draw_raw=True, quantile=0):
    super().__init__(field, interval=interval, reduce=reduce)
    self.graph_values = []
    self.smoothing = smoothing
    self.draw_raw = draw_raw
    self.title = field if title is None else title
    self.quantile = quantile

  def do_value_callback(self, value):
    self.graph_values.append(value)
    #TODO: separate title/path?
    save_plot(self.graph_values, self.title, self.smoothing,
      q=self.quantile, draw_raw=self.draw_raw)

#TODO: histogram per action?
class TensorboardCallback(ValueCallback):
  """Requires TF eager to be enabled
  step_for_step=False is useful for per-episode stats (alongside interval="done")
  """
  def __init__(self, writer, name, interval=None,
       summary_type="scalar", reduce="sum", step_for_step=True):
    super().__init__(interval=interval, reduce=reduce)
    self.writer = writer
    self.step_for_step = step_for_step
    self.name = name

    #TODO: support other types
    s = tf.contrib.summary
    stype = summary_type.lower()
    self.summary_type = (s.scalar if stype == "scalar"
      else s.histogram if stype == "histogram"
      else s.text)

  def get_summary_value(self, value):
    return value

  def do_value_callback(self, value):
    value = self.get_summary_value(value)

    step = self.step_counter if self.step_for_step else self.call_counter
    with self.writer.as_default(), tf.contrib.summary.always_record_summaries():
      if isinstance(value, dict):
        for k, v in value.items():
          self.summary_type(self.name + "/{}".format(k), v, step=step)
      elif isinstance(value, list):
        for i, v in enumerate(value):
          self.summary_type(self.name + "/{}".format(i), v, step=step)
      else:
        self.summary_type(self.name, value, step=step)

#TODO: dont require {} to be present in format str?
class TensorboardFieldCallback(TensorboardCallback):
  """Logs a single field from the callback data"""
  def __init__(self, writer, field, interval=None, name_format="{}",
       summary_type="scalar", reduce="sum", step_for_step=True):
    super().__init__(writer, name_format.format(field), interval=interval,
         summary_type=summary_type, reduce=reduce, step_for_step=step_for_step)

    self.field = field

  def get_value(self, data):
    return data[self.field]

#TODO: for now, weights is a list, but make it a dict so we can name histograms
class TensorboardPGETWeights(TensorboardCallback):
  """Logs PGET weights as histograms"""
  def __init__(self, writer, model_name, interval=None, combine=False, step_for_step=True):
    super().__init__(writer, "{}/weights".format(model_name), interval=interval,
      summary_type="histogram", reduce="last", step_for_step=True)
    self.combine = combine

    #TODO: use model.trainable_variables instead of get_weights()?

  def get_value(self, data):
    return data["agent"].model.get_weights()

  def get_summary_value(self, weights):
    #graph each separately (list of weights)
    if not self.combine:
      return weights
    #combine into one array
    else:
      weights = [w.flatten() for w in weights]
      return numpy.concatenate(weights)

#TODO: move to ml-utils?
def remove_outliers(data, z=2):
  try:
    data = data.numpy()
  except:
    data = np.array(data)
  data = data.flatten()
  return data[abs(data - np.mean(data)) < z * np.std(data)]

#TODO: DRY
class TensorboardPGETTraces(TensorboardCallback):
  """Logs PGET traces as histograms"""
  def __init__(self, writer, model_name, interval=None, combine=False,
      step_for_step=True, outlier_z=2):
    super().__init__(writer, "{}/traces".format(model_name), interval=interval,
    summary_type="histogram", reduce="last", step_for_step=True)
    #TODO: move quantile to tensorboardcallback histogram mode
    self.combine = combine
    self.outlier_z = outlier_z

  def get_value(self, data):
    return data["agent"].traces

  def get_summary_value(self, traces):
    #graph each separately (list of weights)
    if not self.combine:
      return [remove_outliers(t, self.outlier_z) for t in traces]
    #combine into one array
    else:
      traces = [w.flatten() for w in traces]
      return remove_outliers(numpy.concatenate(traces), self.outlier_z)

class TensorboardPGETReward(TensorboardCallback):
  """Logs reward mean/deviation and advantage as scalars"""
  def __init__(self, writer, model_name, interval=None, step_for_step=True):
    super().__init__(writer, "{}/reward".format(model_name), interval=interval,
    summary_type="scalar", reduce="last", step_for_step=True)

  def get_value(self, data):
    agent = data["agent"]
    return {
      "mean": agent.reward_mean,
      "deviation": agent.reward_deviation,
      "advantage": agent.last_advantage
    }

#TODO below
class TensorboardActions(TensorboardFieldCallback):
  def __init__(self, writer, env_name, interval=None, step_for_step=True):
    super().__init__(writer, "action", interval=interval, name_format="{}/" + env_name,
      reduce="mean", step_for_step=step_for_step, summary_type="histogram")

class ValuePrinter(ValueCallback):
  def __init__(self, field, interval=None, reduce="sum", interval_name="Ep"):
    super().__init__(field, interval=interval, reduce=reduce)
    interval_name = interval_name

  def do_callback(self, value):
    print("{} {}: {}".format(self.interval_name, self.call_counter, value))

class ModelSaver(AegisCallback):
  def __init__(self, path, model, interval=1000):
    super().__init__(interval)
    self.path = path
    self.model = model

  def do_callback(self, data):
    print("saving model to {}".format(self.path))
    self.model.save(self.path)
