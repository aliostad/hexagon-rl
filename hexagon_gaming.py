import math
from hexagon import *
import numpy as np


class Move:
  def __init__(self, fromCell, toCell, resources):
    """

    :type fromCell: CellId
    :type toCell: CellId
    :type resources: int
    """
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

  def to_json(self):
    return {
      'roundNo': self.roundNo,
      'ownedCells': map(lambda x: x.to_json(), self.ownedCells)
    }


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


class GameStat:
  def __init__(self, name, rnd, playerStats, finished):
    """

    :type name: str
    :type rnd: int
    :type playerStats: list of PlayerStat
    :param finished: boolean
    """
    self.name = name
    self.round = rnd
    self.playerStats = playerStats
    self.finished = finished


class Player:
  """
  abstract class for Player
  """

  def __init__(self, name):
    self.name = name
    self._started = False
    self.current_game = None

  def start(self, gameName):
    if self._started:
      raise RuntimeError('Player already started')
    self._started = True
    self.current_game = gameName
    return True

  def finish(self):
    """

    :return: dict
    """
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

  def clone(self):
    pass


class Slot:
  def __init__(self, name, players):
    self.name = name
    self.games = []
    self.player_scores = {}
    self.players = players
    for p in players:
      self.player_scores[p.name] = 0

  def add_game_stats(self, gameStat):
    """

    :type GameStat:
    :return:
    """
    self.games.append(gameStat)
    for ps in gameStat.playerStats:
      if ps.playerName not in self.player_scores:
        self.player_scores[ps.playerName] = 0
    self.player_scores[gameStat.playerStats[0].playerName] += 1


class Game:
  def __init__(self, slot, name, players, maxRounds, radius=None, verbose=True, move_shuffle=True):
    """

    :type slot: str
    :type name: str
    :type players: list of Player
    :type radius: int
    """
    self.slot = slot
    self.name = name
    self.players = players
    self.radius = Game.get_optimum_board_size(len(players)) if radius is None else radius
    self.real_players = []
    self._started = False
    self.max_rounds = maxRounds
    self.round_no = 0
    self.board = None
    self.verbose = verbose
    self.move_shuffle = move_shuffle

  def clone(self):
    g = Game(self.slot, self.name,
             map(lambda x: x.clone(), self.players),
             self.radius,
             verbose=False,
             move_shuffle=self.move_shuffle)

    g.real_players = map(lambda x: x.clone(), self.real_players)
    g._started = self._started
    g.board = self.board.clone()
    g.round_no = self.round_no
    return g

  @staticmethod
  def get_optimum_board_size(number_of_players):
    return int(math.sqrt(number_of_players) * 5)

  def _put_players_on_seeds(self):

    '''
    bands = [
      6,
      6 + 2*6,
      6 + 2*6 + 4*6,
      6 + 2*6 + 4*6 + 8*6
    ]

    n_left = len(self.real_players)
    n = 1
    for i in range(len(bands)-1,0, -1):
      if n_left > bands[i]:
        n = i+2
        break

    while True:
    :return:
    '''
    # HACK!!!
    pole = int(2*self.radius/3)-1
    self.board.change_ownership(CellId(-pole, pole), self.real_players[0].name, Cell.MaximumResource)
    self.board.change_ownership(CellId(pole, -pole), self.real_players[1].name, Cell.MaximumResource)
    if len(self.real_players) > 2:
      self.board.change_ownership(CellId(pole, 0), self.real_players[2].name, Cell.MaximumResource)
    if len(self.real_players) > 3:
      self.board.change_ownership(CellId(-pole, 0), self.real_players[3].name, Cell.MaximumResource)
    if len(self.real_players) > 4:
      self.board.change_ownership(CellId(0, pole), self.real_players[4].name, Cell.MaximumResource)
    if len(self.real_players) > 5:
      self.board.change_ownership(CellId(0, -pole), self.real_players[5].name, Cell.MaximumResource)

  def start(self):
    if self._started:
      raise RuntimeError('Game already started')
    self.real_players = filter(lambda player: player.start(self.name), self.players)
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
    idx = np.random.permutation(np.arange(len(self.real_players))) if \
            self.move_shuffle else range(0, len(self.real_players))

    moves = []
    for i in idx:
      p = self.real_players[i]
      try:
        infos = self.board.get_cell_infos_for_player(p.name)
        view = PlayerView(self.round_no, infos)
        if len(infos) > 0:
          mv = p.move(view)
          if mv is not None:
            moves.append((p, mv))
      except Exception as e:
        print('Error in move {} for player {}: {}'.format(self.round_no, p.name, e.message))
    clashes = {}
    for t in moves:
      (p, mv) = t
      if mv.toCell not in clashes:
        clashes[mv.toCell] = []
      if mv.fromCell not in clashes:
        clashes[mv.fromCell] = []
      clashes[mv.toCell].append(p.name)
      clashes[mv.fromCell].append(p.name)
      success, errormsg = self.board.try_transfer(mv)
      if not success and self.verbose:
        p.move_feedback(self.round_no, mv, errormsg)
        print('Move {} from player {} illegal: {}'.format(self.round_no, p.name, errormsg))
    for c in clashes:
      if len(clashes[c]) > 1 and self.verbose:
        print('{} - Clash between {} on {}'.format(self.round_no, ', '.join(clashes[c]), c))
    # OK now increment
    self.board.increment_resources()
    stats = self.get_player_stats()
    isFinished = self.round_no >= self.max_rounds or max(map(lambda x: x.cellsOwned, stats)) == sum(map(lambda x: x.cellsOwned, stats))
    return self.get_player_stats(), isFinished, self._finish_players() if isFinished else {}

  def get_player_stat(self, name):
    infos = self.board.get_cell_infos_for_player(name)
    return PlayerStat(name, self.round_no, len(infos), sum(map(lambda x: x.resources, infos), 0))

  def get_player_stats(self):
    return sorted(map(lambda p: self.get_player_stat(p.name), self.real_players), key=lambda s: s.cellsOwned, reverse=True)

  def _finish_players(self):
    result = {}
    for p in self.real_players:
      result[p.name] = p.finish()
    return result

  def finish(self):
    if not self._started:
      raise RuntimeError('Game not started')
    self._started = False


class GameSnapshot:
  def __init__(self, game, slotName, playerScores):
    """

    :type game: Game
    """
    self.boardSnapshot = game.board.get_snapshot()
    self.stat = GameStat(game.name, game.round_no, game.get_player_stats(), False)
    self.radius = game.board.radius
    self.slotName = slotName
    self.playerScores = playerScores

