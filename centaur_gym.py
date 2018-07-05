from centaur import *
from hexagon_agent import *

class CentaurGym:
  def __init__(self):
    self.players = [
      Aliostad('random05', 0.5),
      Aliostad('random03', 0.3),
      Aliostad('Ali-1'),
      Centaur('centaur', 'zibolon')
    ]
    self.game = None

  def start(self, rounds=1000):
    self.game = Game('my-game', self.players, 8)
    self.game.start()
    for i in range(0, rounds):
      stats, finished = self.game.run_sync()
      for s in stats:
        print(s)
      if finished:
        break

  def finish(self):
    self.game.finish()

if __name__ == '__main__':
  gym = CentaurGym()
  gym.start()
  gym.f