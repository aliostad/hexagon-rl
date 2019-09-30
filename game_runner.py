from hexagon_agent import *
import hexagon_ui_api
import time


class GameRunner:
  def __init__(self, players, slotName):
    '''

    :type players: list of Player
    '''
    self.players = players
    self.game = None
    self.slot = Slot(slotName, players)
    hexagon_ui_api.slot = self.slot

  def run(self, name, games=10, rounds=1000):
    for gid in range(0, games):
      self.game = Game(self.slot.name, str(gid+1), self.players, rounds, 12)
      hexagon_ui_api.games['1'] = self.game
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


if __name__ == '__main__':
    ps = [
        Aliostad('Interpid Ibex', 0.5),
        Aliostad('Jaunty Jackalope', 0.3),
        Aliostad('Karmic Koala'),
        Aliostad('Natty Narwhal', 0.9)]
    runner = GameRunner(ps, '1')
    hexagon_ui_api.run_in_background()
    runner.run('main_game', 100, 300)
    runner.finish()