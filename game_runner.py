from hexagon_agent import *
import time


class GameRunner:
  def __init__(self, slot):
    self.slot = slot
    self.game = None

  def run_many(self, games=10, rounds=1000, radius=None, signaller=None):
    for gid in range(0, games):
      if signaller is not None and signaller.signalled:
        break
      name = str(gid+1)
      self.game = Game(self.slot.name, name, self.slot.players, rounds, radius)
      self.game.start()
      for i in range(0, rounds):
        if signaller is not None and signaller.signalled:
          break
        stats, finished, info = self.game.run_sync()
        time.sleep(0.01)
        if finished:
          gs = GameStat(name, i, stats, finished)
          self.slot.add_game_stats(gs)
          break

  def finish(self):
    self.game.finish()

