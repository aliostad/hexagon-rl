import alpha_zero_general.Game
from hexagon_agent import Aliostad
from hexagon_gaming import Game, Board, Cell, Player
from ai_gym import CentaurAttackProcessor
from numpy import np
from square_grid import *

class AlphaAliostad(Aliostad):

  def __init__(self, name):
    super(Aliostad, self).__init__(name)


class HexagonGame(alpha_zero_general.Game):

  def __init__(self, radius):
    self.radius = radius
    self.rect_width = radius * 2 - (radius % 2)
    self.model_input_shape = (self.rect_width, self.rect_width)
    self.game = None
    self.game_no = 0

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

  def getInitBoard(self):
    self.game_no += 1
    self.game = Game(str(self.game_no),
                     [AlphaAliostad('1'), AlphaAliostad('-1')],
                     self.radius)
    return self._get_board_repr(self.game.board)

  def getNextState(self, board, player, action):
    """

    :type board: ndarray
    :param player: int
    :param action: int
    :return: (ndarray, )
    """
    #  if has no legal move then board does not change
    if action == self.n*self.n:
      return board, -player
    b = Board(self.n)
    b.pieces = np.copy(board)
    move = (int(action/self.n), action%self.n)
    b.execute_move(move, player)
    return (b.pieces, -player)
