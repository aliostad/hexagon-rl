from alpha_zero_general.Game import Game as AlphaGame
from alpha_zero_general.NeuralNet import NeuralNet
from alpha_zero_general.Coach import Coach
from hexagon_agent import Aliostad, UberCell
from hexagon_gaming import Game, Board, Cell, Player
import numpy as np
from square_grid import *
from keras import Model
from keras.layers import Conv2D, Dense, Input, Reshape, Flatten
from keras.optimizers import Adam
import hexagon_ui_api
import os


class PLayerIds:
  Player1 = 1
  Player2 = -1

class PlayerNames:
  Player1 = "1"
  Player2 = "-1"

_board_cache = {}


def sameSign(a, b):
  return (a < 0 and b < 0) or (a > 0 and b > 0)

def oppositeSign(a, b):
  return (a < 0 and b > 0) or (a > 0 and b < 0)

class AlphaAliostad(Aliostad):

  def __init__(self, name):
    Aliostad.__init__ (self, name)
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

def hydrate_board_from_model(a, radius, rect_width):
  """

  :type a: ndarray
  :type radius: int
  :return: Board
  """
  if radius not in _board_cache:
    _board_cache[radius] = Board(radius)
  b = _board_cache[radius].clone()
  for cellId in b.cells:
    hid = GridCellId.fromHexCellId(cellId)
    thid = GridCellId(hid.x, hid.y).transpose(-rect_width / 2, -rect_width / 2)
    value = a[thid.x][thid.y]
    b.change_ownership(cellId, get_player_name_from_resource(value), int(abs(value)))
  return b


class HexagonGame(AlphaGame):
  def __init__(self, radius, verbose=True, debug=False):
    self.radius = radius
    self.rect_width = radius * 2 - (radius % 2)
    self.model_input_shape = (self.rect_width, self.rect_width)
    self.game = None
    self.game_no = 0
    self.NO_LEGAL_MOVE = self.rect_width ** 2
    self.validMovesHistory = []  # for debugging
    self.debug = debug
    self.verbose = verbose

  def getBoardSize(self):
    return self.rect_width, self.rect_width

  def getActionSize(self):
    return (self.rect_width ** 2) + 1  # adding one when no legal move possible

  def _get_board_repr(self, board):
    """

    :param board: Board
    :return: ndarray
    """
    result = np.zeros(self.model_input_shape)

    for c in board.cells.values():
      if c.owner != Cell.NoOwner:
        sign = int(c.owner)
        value = c.resources * sign
        hid = GridCellId.fromHexCellId(c.id)
        thid = hid.transpose(self.rect_width / 2, self.rect_width / 2)
        result[thid.x][thid.y] = value
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
    hid = GridCellId(x, y)
    thid = hid.transpose(-(self.rect_width / 2), -(self.rect_width / 2))
    cellId = thid.to_cell_id()
    cells = game.board.get_cell_infos_for_player(player.name)
    world = Aliostad.build_world(cells)
    if cellId not in world.uberCells:
      print('Not in: {}'.format(cellId))
      return None
    cellFrom = world.uberCells[cellId]
    if cellFrom.canAttackOrExpand:
      return player.getAttack(world, cellId)
    else:
      return player.getBoost(world, cellId)


  def getInitBoard(self):

    self.game_no += 1
    self.game = Game('1',
                     [AlphaAliostad(PlayerNames.Player1), AlphaAliostad(PlayerNames.Player2)],
                     self.radius)
    self.game.start()
    hexagon_ui_api.games['1'] = self.game
    if self.verbose:
      print("New game: {}".format(self.game_no))
    return self._get_board_repr(self.game.board)

  def getNextState(self, board, player, action, executing=False):
    """

    :type board: ndarray
    :param player: int
    :param action: int
    :return: (ndarray, int)
    """
    #  if has no legal move then board does not change
    if action == self.NO_LEGAL_MOVE:
      return board, -player

    hex_board = hydrate_board_from_model(board, self.radius, self.rect_width)
    if executing:
      g = self.game
    else:
      g = self.game.clone()
      g.board = hex_board

    thePlayer = filter(lambda x: x.name == str(player), self.game.real_players)[0]
    move = self.get_move_for_action(g, action, thePlayer)
    if move not None:
      success, msg = g.board.try_transfer(move)

      if player < 0:
        g.board.increment_resources()
        g.round_no += 1
        if self.verbose:
          print('round {}'.format(self.game.round_no))
      if not success:
        print(msg)

    return self._get_board_repr(g.board), -player

  def getValidMoves(self, cannonicalBoard, player):
    """

    :type cannonicalBoard: ndarray
    :param player: int
    :return:
    """
    board = cannonicalBoard*player
    hex_board = hydrate_board_from_model(board, self.radius, self.rect_width)
    g = self.game.clone()
    g.board = hex_board
    cells = g.board.get_cell_infos_for_player(str(player))
    world = Aliostad.build_world(cells)
    result = np.zeros(self.getActionSize())
    if self.debug:
      self.validMovesHistory.append(result)

    # first attack
    for uc in world.uberCells.values():
      hid = GridCellId.fromHexCellId(uc.id)
      thid = GridCellId(hid.x, hid.y).transpose(self.rect_width / 2, self.rect_width / 2)
      idx = thid.x * self.rect_width + thid.y
      if uc.canAttackOrExpand:
        result[idx] = 1
    if result.sum() > 0:
      return result
    elif len(world.uberCells) > 1:
      # boost
      for uc in world.uberCells.values():
        hid = GridCellId.fromHexCellId(uc.id)
        thid = GridCellId(hid.x, hid.y).transpose(self.rect_width / 2, self.rect_width / 2)
        idx = thid.x * self.rect_width + thid.y
        if uc.resources > 2:
          result[idx] = 1
    else:
      result[-1] = 1  # last cell is for NoValidMove
    return result

  def getGameEnded(self, board, player):
    """

    :type board: ndarray
    :type player: int
    :return:
    """

    sameSignCount = 0
    oppositeSignCount = 0
    for v in board.flatten():
      if sameSign(v, player):
        sameSignCount += 1
      if oppositeSign(v, player):
        oppositeSignCount += 1

    if oppositeSignCount > 0 and sameSignCount > 0:
      if self.game.round_no > 200:
        return 1 if sameSignCount > oppositeSignCount else -1
      else:
        return 0
    elif oppositeSignCount > 0:
      return -1
    else:
      return 1



  def getCanonicalForm(self, board, player):
    # return state if player==1, else return -state if player==-1
    return board*player

  def getSymmetries(self, board, pi):
    # mirror, rotational
    assert(len(pi) == self.rect_width**2+1)  # 1 for pass
    pi_board = np.reshape(pi[:-1], (self.rect_width, self.rect_width))
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
    return board.tostring()


class HexagonModel(NeuralNet):

  def __init__(self, game, lr=0.001, batch_size=100, epochs=10):
    """

    :type game: HexagonGame
    """
    input_shape = (game.rect_width, game.rect_width)
    input = Input(shape=input_shape)
    med = Reshape(input_shape + (1, ))(input)
    med = Conv2D(128, (5, 5), padding='same', activation='relu')(med)
    med = Conv2D(64, (3, 3), padding='same', activation='relu')(med)
    med = Conv2D(16, (3, 3), padding='same', activation='relu')(med)
    med = Conv2D(4, (3, 3), padding='same', activation='relu')(med)
    med = Conv2D(1, (3, 3), padding='same', activation='tanh')(med)
    pipe = Flatten()(med)
    pi = Dense(game.getActionSize(), activation='softmax')(pipe)
    v = Dense(1, activation='tanh')(pipe)
    self.model = Model(inputs=[input], outputs=[pi, v])
    self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(lr))
    self.batch_size = batch_size
    self.epochs = epochs

  def train(self, examples):
    input_boards, target_pis, target_vs = list(zip(*examples))
    input_boards = np.asarray(input_boards)
    target_pis = np.asarray(target_pis)
    target_vs = np.asarray(target_vs)
    self.model.fit(x=input_boards, y=[target_pis, target_vs],
                   batch_size=self.batch_size, epochs=self.epochs)
  def predict(self, board):
    """
    board: np array with board
    """
    # preparing input
    board = board[np.newaxis, :, :]

    # run
    pi, v = self.model.predict(board)
    return pi[0], v[0]

  def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(folder):
      print("Checkpoint Directory does not exist! Making directory {}".format(folder))
      os.mkdir(folder)
    else:
      print("Checkpoint Directory exists! ")
    self.model.save_weights(filepath)

  def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
      raise("No model in path '{}'".format(filepath))
    self.model.load_weights(filepath)

class dotdict(dict):
  def __getattr__(self, name):
    return self[name]

if __name__ == '__main__':

  args = dotdict({
    'numIters': 100,
    'numEps': 10,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

  })
  g = HexagonGame(3)
  nnet = HexagonModel(g)
  c = Coach(g, nnet, args)
  hexagon_ui_api.run_in_background()

  if args.load_model:
    print("Load trainExamples from file")
    c.loadTrainExamples()

  c.learn()