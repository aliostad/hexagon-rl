import math
from hexagon import *
import numpy as np

class Move:
  def __init__(self, fromCell, toCell, resources):
    self.fromCell = fromCell
    self.toCell = toCell
    self.resources = resources


class PlayerView:
  def __init__(self, roundNo, ownedCells):
    """

    :type roundNo: int
    :type ownedCells: list of CellInfo
    """
    self.roundNo = roundNo
    self.ownedCells = ownedCells


class PlayerStat:
  def __init__(self, playerName, roundNo, cellsOwned, totalResources):
    """

    :type playerName: str
    :type roundNo: int
    :type cellsOwned: int
    :param totalResources: int
    """
    self.playerName = playerName
    self.roundNo = roundNo
    self.cellsOwned = cellsOwned
    self.totalResources = totalResources

  def __str__(self):
    return '{}: {}\t\t{}\t{}'.format(self.roundNo, self.playerName, self.cellsOwned, self.totalResources)


class Player:
  """
  abstract class for Player
  """

  def __init__(self, name):
    self.name = name
    self._started = False

  def start(self):
    if self._started:
      raise RuntimeError('Player already started')
    self._started = True
    return True

  def finish(self):
    if not self._started:
      raise RuntimeError('Player not started')
    self._started = False

  def move_feedback(self, roundNo, move, error):
    """

    :type roundNo: int
    :type move: Move
    :type error: str
    :return:
    """
    print('Player {} - move error in round {}: {}'.format(self.name, roundNo, error))

  def move(self, playerView):
    """

    :type playerView: PlayerView
    :return: Move
    """
    raise NotImplementedError("move")



class Game:
  def __init__(self, name, players, radius=None):
    """

    :type name: str
    :type players: list of Player
    :type radius: int
    """
    self.name = name
    self.players = players
    self.radius = Game.get_optimum_board_size(len(players)) if radius is None else radius
    self.real_players = []
    self._started = False
    self.round_no = 0
    self.board = None

  @staticmethod
  def get_optimum_board_size(number_of_players):
    return int(math.sqrt(number_of_players) * 5)

  def _put_players_on_seeds(self):
    # HACK!!!
    self.board.change_ownership(CellId(-4, 4), self.real_players[0].name, Cell.MaximumResource)
    self.board.change_ownership(CellId(4, -4), self.real_players[1].name, Cell.MaximumResource)
    if len(self.real_players) > 2:
      self.board.change_ownership(CellId(4, 0), self.real_players[2].name, Cell.MaximumResource)
    if len(self.real_players) > 3:
      self.board.change_ownership(CellId(-4, 0), self.real_players[3].name, Cell.MaximumResource)

  def start(self):
    if self._started:
      raise RuntimeError('Game already started')
    self.real_players = filter(lambda player: player.start(), self.players)
    if len(self.real_players) > 1:
      self._started = True
      self.board = Board(self.radius)
      self._put_players_on_seeds()
    else:
      for p in self.real_players:
        p.finish()
    return self._started

  def run_sync(self):
    self.round_no += 1
    idx = np.random.permutation(np.arange(len(self.real_players)))
    for i in idx:
      p = self.real_players[i]
      moves = []
      try:
        infos = self.board.get_cell_infos_for_player(p.name)
        view = PlayerView(self.round_no, infos)
        if len(infos) > 0:
          mv = p.move(view)
          if mv is not None:
            moves.append((p, mv))
      except Exception as e:
        print('Error in move {} for player {}: {}'.format(self.round_no, p.name, e.message))
      for t in moves:
        (p, mv) = t
        success, errormsg = self.board.try_transfer(mv)
        if not success:
          print('Move {} from player {} illegal: {}'.format(self.round_no, p.name, errormsg))

    # OK now increment
    self.board.increment_resources()
    stats = self.get_player_stats()
    return self.get_player_stats(), max(map(lambda x: x.cellsOwned, stats)) == sum(map(lambda x: x.cellsOwned, stats))

  def get_player_stat(self, name):
    infos = self.board.get_cell_infos_for_player(name)
    return PlayerStat(name, self.round_no, len(infos), sum(map(lambda x: x.resources, infos), 0))

  def get_player_stats(self):
    return sorted(map(lambda p: self.get_player_stat(p.name), self.real_players), key=lambda s: s.cellsOwned, reverse=True)

  def finish(self):
    if not self._started:
      raise RuntimeError('Game not started')
    self._started = False
    for p in self.real_players:
      p.finish()

