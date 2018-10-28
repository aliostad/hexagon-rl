import alpha_zero_general.Game
from hexagon_agent import Aliostad, UberCell
from hexagon_gaming import Game, Board, Cell, Player
from numpy import np
from square_grid import *

class PLayerIds:
  Player1 = 1
  Player2 = -1

class PlayerNames:
  Player1 = "1"
  Player2 = "2"

_board_cache = {}


def sameSign(a, b):
  return (a < 0 and b < 0) or (a > 0 and b > 0)


class AlphaAliostad(Aliostad):

  def __init__(self, name):
    super(Aliostad, self).__init__(name)
    self.id = 'N/A'


def get_player_name_from_resource(value):
  """

  :type value: int
  :return: str
  """
  if value == 0:
    return Cell.NoOwner
  elif value > 0:
    return PlayerNames.Player1
  else:
    return PlayerNames.Player2

def hydrate_board_from_model(a, radius):
  """

  :type a: ndarray
  :type radius: int
  :return: Board
  """
  if radius not in _board_cache:
    _board_cache[radius] = Board(radius)
  b = _board_cache[radius].clone()
  for cellId in b.cells:
    thid = GridCellId.fromHexCellId(cellId)
    value = a[thid.x][thid.y]
    b.change_ownership(cellId, get_player_name_from_resource(value), abs(value))
  return b


class HexagonGame(alpha_zero_general.Game):

  def __init__(self, radius):
    self.radius = radius
    self.rect_width = radius * 2 - (radius % 2)
    self.model_input_shape = (self.rect_width, self.rect_width)
    self.game = None
    self.game_no = 0
    self.NO_LEGAL_MOVE = self.rect_width ** 2

  def getBoardSize(self):
    return self.rect_width, self.rect_width

  def getActionSize(self):
    return (self.rect_width ** 2) + 1  # adding one when no legal move possible

  def _get_board_repr(self, board):
    """

    :param board: Board
    :return: ndarray
    """
    result = np.array(self.model_input_shape)

    for c in board.cells.values():
      if c.owner != Cell.NoOwner:
        sign = int(c.owner)
        value = c.resources * sign
        hid = GridCellId.fromHexCellId(cid)
        thid = hid.transpose(self.spatial_input[0] / 2, self.spatial_input[1] / 2)
        hector[thid.x][thid.y] = value
    return result

  def get_move_for_action(self, game, action, player):
    """

    :type game: Game
    :type action: int
    :type player: AlphaAliostad
    :return:
    """
    y = action % self.rect_width
    x = action / self.rect_width
    thid = GridCellId(x, y)
    cellId = thid.to_cell_id()
    cells = game.board.get_cell_infos_for_player(player.name)
    world = Aliostad.build_world(cells)
    cellFrom = world.uberCells[cells]
    if cellFrom.canAttackOrExpand:
      return player.getAttack(world, cellId)
    else:
      return player.getBoost(world, cellId)


  def getInitBoard(self):
    self.game_no += 1
    self.game = Game(str(self.game_no),
                     [AlphaAliostad(PlayerNames.Player1), AlphaAliostad(PlayerNames.Player2)],
                     self.radius)
    return self._get_board_repr(self.game.board)

  def getNextState(self, board, player, action):
    """

    :type board: ndarray
    :param player: int
    :param action: int
    :return: (ndarray, int)
    """
    #  if has no legal move then board does not change
    if action == self.NO_LEGAL_MOVE:
      return board, -player

    hex_board = hydrate_board_from_model(board, self.radius)
    g = self.game.clone()
    g.board = hex_board
    thePlayer = filter(lambda x: x.name == str(player), self.game.real_players)[0]
    move = self.get_move_for_action(g, action, thePlayer)
    succss, msg = g.board.try_transfer(move)
    if not succss:
      raise Exception(msg)

    return self._get_board_repr(g.board), -player

  def getValidMoves(self, board, player):
    """

    :type board: ndarray
    :param player: int
    :return:
    """
    hex_board = hydrate_board_from_model(board, self.radius)
    g = self.game.clone()
    g.board = hex_board
    cells = g.board.get_cell_infos_for_player(str(player))
    world = Aliostad.build_world(cells)
    result = np.array(self.getActionSize())
    result[-1] = 1  # last cell is for NoValidMove
    if len(world.uberCells) < 2:  # no legal move possible
      return result
    for uc in world.uberCells.values():
      thid = GridCellId.fromHexCellId(uc.id)
      idx = thid.x * self.rect_width + thid.y
      if uc.canAttackOrExpand or uc.resources > 2:
        result[idx] = 1
    return result

  def getGameEnded(self, board, player):
    """

    :type board: ndarray
    :type player: int
    :return:
    """
    for v in board.flatten():
      if sameSign(v, player):
        return True
    return False

  def getCanonicalForm(self, board, player):
    # return state if player==1, else return -state if player==-1
    return player*board

  def getSymmetries(self, board, pi):
    # mirror, rotational
    assert(len(pi) == self.n**2+1)  # 1 for pass
    pi_board = np.reshape(pi[:-1], (self.n, self.n))
    l = []

    for i in range(1, 5):
      for j in [True, False]:
        newB = np.rot90(board, i)
        newPi = np.rot90(pi_board, i)
        if j:
          newB = np.fliplr(newB)
          newPi = np.fliplr(newPi)
        l += [(newB, list(newPi.ravel()) + [pi[-1]])]
    return l

  def stringRepresentation(self, board):
    # 8x8 numpy array (canonical board)
    return board.tostring()

