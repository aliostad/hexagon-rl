from hexagon_agent import *
import time


class GameRunner:
  def __init__(self, players, slotName):
    '''

    :type players: list of Player
    '''
    self.players = players
    self.game = None
    self.slot = Slot(slotName, players)

  def run_many(self, games=10, rounds=1000):
    for gid in range(0, games):
      name = str(gid+1)
      self.game = Game(self.slot.name, name, self.players, rounds, 12)
      self.game.start()
      for i in range(0, rounds):
        stats, finished, info = self.game.run_sync()
        time.sleep(0.05)
        if finished:
          gs = GameStat(name, i, stats, finished)
          self.slot.add_game_stats(gs)
          break

  def finish(self):
    self.game.finish()

