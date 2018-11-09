from hexagon_agent import *
import hexagon_ui_api
import time

class CentaurGym:
  def __init__(self):
    self.players = [
      Aliostad('Interpid Ibex', 0.5),
      Aliostad('Jaunty Jackalope', 0.3),
      Aliostad('Karmic Koala'),
      Aliostad('Natty Narwhal', 0.9)
    ]
    self.game = None

  def start(self, games=10, rounds=1000):
    for gid in range(0, games):
      self.game = Game('my-game', self.players, 6)
      hexagon_ui_api.games['1'] = gym.game
      self.game.start()
      for i in range(0, rounds):
        stats, finished, info = self.game.run_sync()
        time.sleep(0.05)
        if finished:
          break

  def finish(self):
    self.game.finish()

if __name__ == '__main__':
  gym = CentaurGym()
  hexagon_ui_api.run_in_background()
  gym.start()
  gym.finish()