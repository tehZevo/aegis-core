import time

class Controller:
  def __init__(self, engine, niceness=0.01):
    self.niceness = niceness
    self.engine = engine
    self.reward = 0
    self.state = None #TODO: initial value?

  def loop(self):
    while True:
      r = self.reward
      self.reward = 0
      starttime = time.time()
      self.state = self.engine.update(r)
      #sleep time equal to update time * niceness
      dt = time.time() - starttime
      if self.niceness >= 0:
        time.sleep(dt * self.niceness)
      else:
        time.sleep(-self.niceness)
