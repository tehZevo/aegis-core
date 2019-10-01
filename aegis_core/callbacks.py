#WIP
class WeightVisualizer():
  def __init__(self, path, interval=1000):
    self.steps_since_save = 0
    self.save_interval = interval
    self.save_path = path

  def __call__(self, engine):
    self.steps_since_save += 1
    if self.steps_since_save > self.save_interval:
      print("plotting weights to {}".format(self.save_path))
      viz_weights(engine.model.get_weights(), self.save_path + ".png")
      self.steps_since_save = 0

class ModelSaver():
  def __init__(self, path, interval=1000):
    self.steps_since_save = 0
    self.save_interval = interval
    self.save_path = path

  def __call__(self):
    self.steps_since_save += 1
    if self.steps_since_save > self.save_interval:
      print("saving model to {}".format(self.save_path))
      self.model.save(self.save_path)
      self.steps_since_save = 0
