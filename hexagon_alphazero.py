from alpha_zero_general.Game import Game as AlphaGame
from alpha_zero_general.NeuralNet import NeuralNet
from alpha_zero_general.Coach import Coach
from alpha_zero_general.Arena import Arena
from alpha_zero_general.MCTS import MCTS
from hexagon_agent import Aliostad, UberCell
from hexagon_gaming import Game, Board, Cell, Player, Move, CellId
import numpy as np
from square_grid import *
from keras import Model
from keras.layers import Conv2D, Dense, Input, Reshape, Flatten
from keras.optimizers import Adam
import hexagon_ui_api
import os
import sys


class PlayerIds:
  Player1 = 1
  Player2 = -1

class PlayerNames:
  Player1 = "1"
  Player2 = "-1"

def sameSign(a, b):
  return (a < 0 and b < 0) or (a > 0 and b > 0)

def oppositeSign(a, b):
  return (a < 0 and b > 0) or (a > 0 and b < 0)

class PlayerNameMapper:
  def __init__(self):
    self.hex_to_alpha = {}
    self.alpha_to_hex = {}
    
  def register_player_name(self, hex_name, alpha_name):
    self.alpha_to_hex[alpha_name] = hex_name
    self.hex_to_alpha[hex_name] = alpha_name
  
  def get_hex_name(self, alpha_name):
    return self.alpha_to_hex[alpha_name]
  
  def get_alpha_name(self, hex_name):
    return self.hex_to_alpha[hex_name]

_player_name_mapper = PlayerNameMapper()


def get_player_name_from_resource(value):
  """

  :type value: int
  :return: str
  """
  if value == 0:
    return Cell.NoOwner
  elif value > 0:
    return _player_name_mapper.get_hex_name(PlayerNames.Player1)
  else:
    return _player_name_mapper.get_hex_name(PlayerNames.Player2)

def hydrate_board_from_model(a, radius, rect_width):
  """

  :type a: ndarray
  :type radius: int
  :return: Board
  """
  b = Board(radius)
  for cellId in b.cells:
    thid = get_thid_from_cellId(cellId, rect_width)
    value = a[thid.y][thid.x]
    b.change_ownership(cellId, get_player_name_from_resource(value), int(abs(value)))
  return b


class HexagonGame(AlphaGame):
  def __init__(self, radius, verbose=True, debug=False, allValidMovesPlayer=None):
    self.radius = radius
    self.rect_width = radius * 2 - (radius % 2)
    self.model_input_shape = (self.rect_width, self.rect_width)
    self.game = None
    self.game_no = 0
    self.NO_LEGAL_MOVE = self.rect_width ** 2
    self.validMovesHistory = []  # for debugging
    self.debug = debug
    self.verbose = verbose
    self.all_valid_moves_player = allValidMovesPlayer

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
        sign = int(_player_name_mapper.get_alpha_name(c.owner))
        value = c.resources * sign
        thid = get_thid_from_cellId(c.id, self.rect_width)
        result[thid.y][thid.x] = value
    return result

  def get_move_for_action(self, board, action, player):
    """

    :type board: Board
    :type action: int
    :type player: AlphaAliostad
    :return:
    """
    cellId = get_cellId_from_index(action, self.rect_width)
    cells = board.get_cell_infos_for_player(player.name)
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
                     [Aliostad(_player_name_mapper.get_hex_name(PlayerNames.Player1)), 
                      Aliostad(_player_name_mapper.get_hex_name(PlayerNames.Player2))],
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
      b = self.game.board
    else:
      b = hex_board

    thePlayer = filter(lambda x: x.name == _player_name_mapper.get_hex_name(str(player)), self.game.real_players)[0]
    move = self.get_move_for_action(b, action, thePlayer)
    if move is not None:
      success, msg = b.try_transfer(move)
      if player < 0:
        b.increment_resources()
        if executing:
          self.game.round_no += 1
      if not success:
        print(msg)

    return self._get_board_repr(b), -player

  def getValidMoves(self, cannonicalBoard, player, realPlayer=None):
    """

    :type cannonicalBoard: ndarray
    :param player: int
    :return:
    """
    board = cannonicalBoard
    hex_board = hydrate_board_from_model(board, self.radius, self.rect_width)
    cells = hex_board.get_cell_infos_for_player(_player_name_mapper.get_hex_name(str(player)))
    world = Aliostad.build_world(cells)
    result = np.zeros(self.getActionSize())
    if self.debug:
      self.validMovesHistory.append(result)

    if realPlayer is not None and realPlayer == self.all_valid_moves_player:  # send all cells includes both boost and attack
      for uc in world.uberCells.values():
        idx = get_index_from_cellId(uc.id, self.rect_width)
        if uc.canAttackOrExpand or uc.resources > 0:
          result[idx] = 1
      result[-1] = 1  # anyway
    else:
      # first attack
      for uc in world.uberCells.values():
        idx = get_index_from_cellId(uc.id, self.rect_width)
        if uc.canAttackOrExpand:
          result[idx] = 1
      if result.sum() > 0:
        return result
      elif len(world.uberCells) > 1:
        # boost
        for uc in world.uberCells.values():
          idx = get_index_from_cellId(uc.id, self.rect_width)
          if uc.resources > 2:
            result[idx] = 1
    if result.sum() == 0:
      result[-1] = 1  # last cell is for NoValidMove
    return result

  def getGameEnded(self, board, player):
    """

    :type board: ndarray
    :type player: int
    :return:
    """
    result = 0
    sameSignCount = 0
    oppositeSignCount = 0
    sameSignSum = 0
    oppositeSignSum = 0
    for v in board.flatten():
      if sameSign(v, player):
        sameSignCount += 1
        sameSignSum += abs(v)
      if oppositeSign(v, player):
        oppositeSignCount += 1
        oppositeSignSum += v

    if oppositeSignCount > 0 and sameSignCount > 0:
      if self.game.round_no >= 200:
        if abs(sameSignCount - oppositeSignCount) == 1:
          result = 0.1  # draw
        else:
          result = 1 if sameSignCount > oppositeSignCount else -1
      else:
        result = 0
    elif oppositeSignCount > 0:
      result = -1
    else:
      result = 1
    texts = {0.1: 'drew', 1: 'won', -1: 'lost'}
    if result != 0:
      print('Player {} {} !'.format(player, texts[result]))
    return result

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

  def __init__(self, game, lr=0.003, batch_size=100, epochs=100):
    """

    :type game: HexagonGame
    """
    input_shape = (game.rect_width, game.rect_width)
    input = Input(shape=input_shape, name='board_input')
    med = Reshape(input_shape + (1, ))(input)
    med = Conv2D(128, (5, 5), padding='same', activation='relu', name='5x5-128')(med)
    med = Conv2D(64, (3, 3), padding='same', activation='relu', name='3x3-64')(med)
    med = Conv2D(16, (3, 3), padding='same', activation='relu', name='3x3-16')(med)
    med = Conv2D(4, (3, 3), padding='same', activation='relu', name='3x3-4')(med)
    med = Conv2D(1, (3, 3), padding='same', activation='tanh', name='3x3-1')(med)
    pipe = Flatten()(med)
    pi = Dense(game.getActionSize(), activation='softmax', name='out_pi')(pipe)
    v = Dense(1, activation='tanh', name='out_v')(pipe)
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

class AliostadPlayer:

  def __init__(self, game):
    """

    :type game: HexagonGame
    """
    self.game = game
    self.aliostad = Aliostad('aliostad')

  def play(self, board):
    """

    :type board: ndarray
    :return:
    """
    hex_board = hydrate_board_from_model(board, self.game.game.radius, self.game.rect_width)
    cells = hex_board.get_cell_infos_for_player(self.aliostad.name)
    world = Aliostad.build_world(cells)
    move = self.aliostad.movex(world)
    if move is None:
      return self.game.rect_width**2  # no valid move
    cid = move.fromCell
    idx = get_index_from_cellId(cid, self.game.rect_width)
    return idx

class CentaurPlayer:
  
  def __init__(self, game, nnet, args):
    """
    
    :type game: HexagonGame
    :type nnet: HexagonModel
    """
    self.game = game
    self.nnet = nnet
    self.mcts = MCTS(game, nnet, args)
  
  def play(self, board):
    """
    
    :type board: ndarray
    :return: 
    """
    return np.argmax(self.mcts.getActionProb(board, temp=0))
    
    
if __name__ == '__main__':

  args = dotdict({
    'numIters': 20,
    'numEps': 10,
    'tempThreshold': 5,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 3,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/', 'temp.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'radius': 4
  })

  train = True
  test = False
  
  if len(sys.argv) > 1 and sys.argv[1] == 'test':
    train = False
    test = True

  g = HexagonGame(radius=args.radius)
  model = HexagonModel(g)

  hexagon_ui_api.run_in_background()

  if train:
    
    _player_name_mapper.register_player_name('alpha1', PlayerNames.Player1)
    _player_name_mapper.register_player_name('alpha2', PlayerNames.Player2)

    c = Coach(g, model, args)
  
    if args.load_model:
      print("Load trainExamples from file")
      c.loadTrainExamples()
  
    c.learn()
  
  if test:
    _player_name_mapper.register_player_name('aliostad', PlayerNames.Player1)
    _player_name_mapper.register_player_name('centaur', PlayerNames.Player2)
    g.all_valid_moves_player = PlayerIds.Player1

    model.load_checkpoint('temp', 'best.pth.tar')
    aliostad = AliostadPlayer(g)
    centaur = CentaurPlayer(g, model, args)
    
    arena = Arena(aliostad.play, centaur.play, g)
    print(arena.playGames(20, verbose=False))