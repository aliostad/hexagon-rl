from centaur import *
from hexagon_gaming import *
from hexagon_agent import *

class CentaurGym:
  def __init__(self):
    self.players = [
      Aliostad('Ali-1'),
      Aliostad('Ali-2'),
      Aliostad('Ali-3'),
      Centaur('centaur', 'zibolon')
    ]

  def start(self, rounds=1000):
    game = Game('my-game', self.players, 8)
    game.start()
    for i in range(0, rounds):
      stats = game.run_sync()
      for s in stats:
        print(s)


if __name__ == '__main__':
  gym = CentaurGym()
  gym.start()
