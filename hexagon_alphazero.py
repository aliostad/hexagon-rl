from alpha_zero_general.Game import Game as AlphaGame
from alpha_zero_general.NeuralNet import NeuralNet
from alpha_zero_general.Coach import Coach
from alpha_zero_general.Arena import Arena
from alpha_zero_general.MCTS import MCTS
from hexagon_agent import Aliostad, UberCell, World
from hexagon_gaming import Game, Board, Cell, Player, Move, CellId
from ppo import *
import numpy as np
from square_grid import *
from keras import Model, Sequential
from keras.layers import Conv2D, Dense, Input, Reshape, Flatten, concatenate
from keras.optimizers import Adam
import hexagon_ui_api
import os
import sys
import argparse

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
  def __init__(self, radius, verbose=True, debug=False, allValidMovesPlayer=None,
               intelligent_resource_actor=None, max_rounds=None, resource_quantisation=1,
               quantization_proportion=0.7):
    """

    :type radius: int
    :type verbose: bool
    :type debug: bool
    :type allValidMovesPlayer: bool
    :type intelligent_resource_actor: PPOAgent
    :type resource_quantisation: int
    """
    self.radius = radius
    self.rect_width = radius * 2 - (radius % 2)
    self.model_input_shape = (self.rect_width, self.rect_width)
    self.game = None
    self.game_no = 0
    self.NO_LEGAL_MOVE = self.rect_width ** 2 * resource_quantisation
    self.validMovesHistory = []  # for debugging
    self.debug = debug
    self.verbose = verbose
    self.all_valid_moves_player = allValidMovesPlayer
    self.previous_allowed_boost = {}
    self.intelligent_resource_actor = intelligent_resource_actor
    self.max_rounds = (radius**2 * 10) if max_rounds is None else max_rounds
    self.intelligent_resource_players = {}
    self.resource_quantisation = resource_quantisation
    self.resource_quant_for_player = {1: quantization_proportion, -1:quantization_proportion}
    self.episode_number = 0

  def getBoardSize(self):
    return self.rect_width, self.rect_width

  def getActionSize(self):
    return (self.rect_width ** 2 * self.resource_quantisation) + 1  # adding one when no legal move possible

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

  def get_move_for_action(self, board, action, hex_player, realPlayerId):
    """

    :type board: Board
    :type action: int
    :type player: Aliostad
    :return: Move, World
    """
    if action == self.NO_LEGAL_MOVE:  # means no valid actions possible
      return None, None, None
    reward = None
    cellId = get_cellId_from_index(action, self.rect_width, self.resource_quantisation)
    cells = board.get_cell_infos_for_player(hex_player.name)
    world = Aliostad.build_world(cells)
    if cellId not in world.uberCells:
      print('Not in: {}'.format(cellId))
      raise Exception('woooowww... what do you think you are doing?!')
    cellFrom = world.uberCells[cellId]
    if cellFrom.canAttackOrExpand:
      move = hex_player.getAttack(world, cellId)
    else:
      move = hex_player.getBoost(world, cellId)
    toResources = world.worldmap[move.toCell]  # negative if attack
    is_attack = False if move.toCell in world.uberCells else True

    if self.intelligent_resource_actor and realPlayerId in self.intelligent_resource_players:
      proportion = self.intelligent_resource_actor.forward(self.extract_resource_feature(move, world))
      move, isValid = self._get_resource_isValid(move, world, proportion)
      reward = 0 if isValid else -5
    if self.resource_quantisation > 1:
      proportion = self.resource_quant_for_player[realPlayerId] if \
         realPlayerId in self.resource_quant_for_player else \
        ((action % self.resource_quantisation)+ 1) / (self.resource_quantisation + 1.)
      if is_attack:
        move.resources = abs(toResources) + max(int(proportion * (cellFrom.resources + toResources)), 1)
      else:
        move.resources = max(int(proportion * cellFrom.resources), 1)
    return move, world, reward


  def getInitBoard(self):

    self.game_no += 1
    players = [Aliostad(_player_name_mapper.get_hex_name(PlayerNames.Player1)),
                      Aliostad(_player_name_mapper.get_hex_name(PlayerNames.Player2))]
    np.random.shuffle(players)
    self.episode_number += 1
    self.game = Game('1', str(self.episode_number), players, self.radius, verbose=self.verbose)
    self.game.start()
    hexagon_ui_api.games['1'] = self.game
    if self.verbose:
      print("New game: {}".format(self.game_no))
    return self._get_board_repr(self.game.board)

  def _is_resource_amount_valid(self, move, world):
    """

    :type move: Move
    :type world: World
    :return:
    """
    if move.resources <= 0:
      return False
    fromCell = world.uberCells[move.fromCell]
    if fromCell.resources <= move.resources:
      return False
    toResources = world.worldmap[move.toCell]
    if move.toCell not in world.uberCells:
      # attack
      if abs(toResources) >= move.resources:
        return False
    return True

  def extract_resource_feature(self, move, world):
    """
    Generates a feature vector for resource. Indices:
      0  - resource of the from cell
      1  - sum of friendly resources
      2  - min of the friendly resources
      3  - max of the friendly resources
      4  - sum of enemy resources
      5  - min of the enemy resources
      6  - max of the enemy resources
      7  - resource of the to cell
      8  - sum of friendly resources
      9  - min of the friendly resources
      10 - max of the friendly resources
      11 - sum of enemy resources
      12 - min of the enemy resources
      13 - max of the enemy resources
    :type move: Move
    :type world: World
    :return:
    """
    def safe_sum(iterable):
      if any(iterable):
        return sum(iterable)
      else:
        return 0
    def safe_min(iterable):
      if any(iterable):
        return np.min(np.array(iterable))
      else:
        return 0
    def safe_max(iterable):
      if any(iterable):
        return np.max(np.array(iterable))
      else:
        return 0

    fromCell = world.uberCells[move.fromCell]
    if fromCell.resources <= move.resources:
      return False
    toResources = world.worldmap[move.toCell]
    vector = [0 for _ in range(0, 14)]
    vector[0] = fromCell.resources * 1.
    is_attack = False if move.toCell in world.uberCells else True
    sign = -1. if is_attack else 1.
    vector[1] = safe_sum(map(lambda x: x.resources, fromCell.owns)) * 1.
    vector[2] = safe_min(map(lambda x: x.resources, fromCell.owns)) * 1.
    vector[3] = safe_max(map(lambda x: x.resources, fromCell.owns)) * 1.
    vector[5] = -safe_sum(map(lambda x: x.resources, fromCell.enemies)) * 1.
    vector[5] = -safe_min(map(lambda x: x.resources, fromCell.enemies)) * 1.
    vector[6] = -safe_max(map(lambda x: x.resources, fromCell.enemies)) * 1.
    vector[7] = sign * toResources
    return np.array(vector)

  def _get_resource_isValid(self, move, world, proportion):
    """

    :type move: Move
    :type world: World
    :type proportion: float
    :return: (int, bool)
    """
    if proportion > 1. or proportion < 0:
      return move, False

    is_attack = move.toCell not in world.uberCells
    fromCell = world.uberCells[move.fromCell]
    if is_attack:
      diff = fromCell.resources + world.worldmap[move.toCell]
      top_up = int(diff * proportion)
      if top_up == 0:
        top_up = 1
      if top_up == diff:
        top_up -= 1
      move.resources = abs(world.worldmap[move.toCell]) + top_up
    else:
      amount = int(proportion * fromCell.resources)
      if amount == 0:
        amount = 1
      if amount == fromCell.resources:
        amount -= 1
      move.resources = amount
    return move, True

  def getNextState(self, board, player, action, executing=False, realPlayer=None):
    """

    :type board: ndarray
    :param player: int
    :param action: int
    :return: (ndarray, int)
    """
    if realPlayer is None:
      realPlayer = player
    #  if has no legal move then board does not change
    if action == self.NO_LEGAL_MOVE:
      return board, -player

    hex_board = hydrate_board_from_model(board, self.radius, self.rect_width)
    if executing:
      b = self.game.board
    else:
      b = hex_board

    hex_player = filter(lambda x: x.name == _player_name_mapper.get_hex_name(str(player)), self.game.real_players)[0]
    move, world, reward = self.get_move_for_action(b, action, hex_player, realPlayer)

    if move is not None:
      success, msg = b.try_transfer(move)
      if player < 0:
        b.increment_resources()
        if executing:
          self.game.round_no += 1
      if not success:
        print(msg)

    newBoard = self._get_board_repr(b)
    if reward is not None:
      if executing and player == PlayerIds.Player1:
        self.intelligent_resource_actor.step += 1
      result = self.getGameEnded(newBoard, realPlayer, not executing)
      if result == 0:
        self.intelligent_resource_actor.backward(reward, False, realPlayer)
      elif executing and result == 1:
        self.intelligent_resource_actor.backward(reward + 1000, True, realPlayer)
        self.intelligent_resource_actor.backward(reward - 1000, True, -realPlayer)
      elif executing and result == -1:
        self.intelligent_resource_actor.backward(reward - 1000, True, realPlayer)
        self.intelligent_resource_actor.backward(reward + 1000, True, -realPlayer)
      elif executing and result == 0.1:
        self.intelligent_resource_actor.backward(reward + 100, True, realPlayer)
        self.intelligent_resource_actor.backward(reward + 100, True, -realPlayer)
    return newBoard, -player

  def getValidMoves(self, cannonicalBoard, player, realPlayer=None):
    """

    :type cannonicalBoard: ndarray
    :param player: int
    :return:
    """
    hex_board = hydrate_board_from_model(cannonicalBoard, self.radius, self.rect_width)
    cells = hex_board.get_cell_infos_for_player(_player_name_mapper.get_hex_name(str(player)))
    world = Aliostad.build_world(cells)
    result = np.zeros(self.getActionSize())
    include_attack = True
    include_boost = False
    include_no_legal_move = False
    if realPlayer is not None:
      if realPlayer not in self.previous_allowed_boost:
        self.previous_allowed_boost[realPlayer] = True
      if not self.previous_allowed_boost[realPlayer]:
        include_boost = True
      self.previous_allowed_boost[realPlayer] = not self.previous_allowed_boost[realPlayer]
      if realPlayer == self.all_valid_moves_player:
        include_boost = True
        include_no_legal_move = True

    if len(world.uberCells) < 2:
      include_boost = False

    for uc in world.uberCells.values():
      idx = get_index_from_cellId(uc.id, self.rect_width, self.resource_quantisation)
      if (uc.canAttackOrExpand and include_attack) or (uc.resources > 1 and include_boost):
        for i in range(0, self.resource_quantisation):
          result[idx + i] = 1
    if result.sum() == 0 or include_no_legal_move:
      result[-1] = 1  # last cell is for NoValidMove
    return result

  def getGameEnded(self, board, player, askingForAFriend=True):
    """

    :type board: ndarray
    :type player: int
    :param askingForAFriend: whether it is hypothetical question for a tree search (asking for a friend)
            or really asking if the game ended in which case 'is NOT asking for a friend'
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
      if self.game.round_no >= self.max_rounds:
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
    if result != 0 and not askingForAFriend:
      print('Player {} {} !'.format(player, texts[result]))
    return result

  def getCanonicalForm(self, board, player):
    # return state if player==1, else return -state if player==-1
    return board*player

  def getSymmetries(self, board, pi):
    # mirror, rotational
    assert(len(pi) == self.rect_width**2*self.resource_quantisation+1)  # 1 for pass
    pi_board = np.reshape(pi[:-1], (self.rect_width, self.rect_width, self.resource_quantisation))

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
      print ("No model in path '{}'".format(filepath))
    else:
      self.model.load_weights(filepath)


class HexagonAlternativeModel(HexagonModel):
  def __init__(self, game, lr=0.003, batch_size=100, epochs=100):
    """

    :type game: HexagonGame
    """
    input_shape = (game.rect_width, game.rect_width)
    input = Input(shape=input_shape, name='board_input')
    med = Reshape(input_shape + (1, ))(input)
    med = Conv2D(128, (5, 5), padding='same', activation='relu', name='5x5-128')(med)
    med = Conv2D(64, (3, 3), padding='same', activation='relu', name='3x3-64')(med)
    med = Flatten()(med)
    med = Dense(64, activation='relu', name='dense64_relu')(med)
    pipe = Dense(64, activation='tanh', name='dense64_tanh')(med)
    pi = Dense(game.getActionSize(), activation='softmax', name='out_pi')(pipe)
    v = Dense(1, activation='tanh', name='out_v')(pipe)
    self.model = Model(inputs=[input], outputs=[pi, v])
    self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(lr))
    self.batch_size = batch_size
    self.epochs = epochs


class HexagonFlatModel(HexagonModel):
  def __init__(self, game, lr=0.003, batch_size=100, epochs=100):
    """

    :type game: HexagonGame
    """
    input_shape = (game.rect_width, game.rect_width)
    input = Input(shape=input_shape, name='board_input')
    med = Flatten()(input)
    med = Dense(128, activation='relu', name='dense64_relu')(med)
    med = Dense(64, activation='tanh', name='dense64_tanh')(med)
    pipe = Dense(64, activation='relu', name='dense64_relu2')(med)
    pi = Dense(game.getActionSize(), activation='softmax', name='out_pi')(pipe)
    v = Dense(1, activation='tanh', name='out_v')(pipe)
    self.model = Model(inputs=[input], outputs=[pi, v])
    self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(lr))
    self.batch_size = batch_size
    self.epochs = epochs

class ResourceModel:
  def __init__(self, modelName='Attack_model_params.h5f', lr=0.001):
    self.modelName = modelName if modelName is not None else 'Attack_model_params.h5f' + str(r.uniform(0, 10000000))
    state_input_shape = (14, )
    action_shape = (1,)

    state_input = Input(shape=state_input_shape, name='state_input')
    advantage = Input(shape=(1,), name='advantage')
    old_prediction = Input(action_shape, name='old_prediction')
    samba = Dense(256, activation='relu', name='actor_dense_0')(state_input)
    samba = Dense(64, activation='relu', name='actor_dense_1')(samba)
    actor_output = Dense(1, activation='tanh', name='actor_output')(samba)
    model = Model(inputs=[state_input, advantage, old_prediction], outputs=[actor_output])
    model.compile(optimizer=Adam(lr=lr),
                  loss=[proximal_policy_optimization_loss_continuous(
                    advantage=advantage,
                    old_prediction=old_prediction)])
    print(model.summary())
    self.model = model
    critic_input = Input(state_input_shape, name='critic_state_input')
    critic_path = Dense(256, activation='relu', name='critic_dense_0')(critic_input)
    critic_path = Dense(256, activation='relu', name='critic_dense_1')(critic_path)
    critic_out = Dense(1, name='critic_output')(critic_path)
    critic = Model(inputs=[critic_input], outputs=[critic_out])
    critic.compile(optimizer=Adam(lr=lr), loss='mse')
    self.critic = critic
    print(critic.summary())

class DotDict(dict):
  def __getattr__(self, name):
    return self[name]

class AliostadPlayer:

  def __init__(self, game, name, am_i_second_player=False, quantization_proportion=0.7):
    """

    :type game: HexagonGame
    :type name: str
    """
    self.game = game
    self.aliostad = Aliostad(name, randomBoostFactor=None)
    self.am_i_second_player = am_i_second_player
    self.quantization_proportion = quantization_proportion

  def play(self, board):
    """

    :type board: ndarray
    :return:
    """
    if self.am_i_second_player:
      board = board * -1
    hex_board = hydrate_board_from_model(board, self.game.game.radius, self.game.rect_width)
    cells = hex_board.get_cell_infos_for_player(self.aliostad.name)
    world = Aliostad.build_world(cells)
    move = self.aliostad.movex(world)
    if move is None:
      return self.game.NO_LEGAL_MOVE  # no valid move
    cid = move.fromCell
    idx = get_index_from_cellId(cid, self.game.rect_width, self.game.resource_quantisation)
    if self.game.resource_quantisation > 1:
      idx += int(self.quantization_proportion * (self.game.resource_quantisation+1))
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
    xx
    :type board: ndarray
    :return: 
    """
    return np.argmax(self.mcts.getActionProb(board, temp=0))

class RandomPlayer:

  def __init__(self, game, playerId):
    """,

    :type game: HexagonGame
    """
    self.game = game
    self.id = playerId

  def play(self, board):
    """
    xx
    :type board: ndarray
    :return:
    """
    choices = []
    valids = self.game.getValidMoves(self.game.getCanonicalForm(board, self.id), self.id, self.id)
    for idx, v in enumerate(valids):
      if v == 1:
        choices.append(idx)
    return np.random.choice(choices)

def menu():
  parser = argparse.ArgumentParser()
  parser.add_argument('what', help="what to do")
  parser.add_argument('--p1', '-p', help="player1: r for random, a for Aliostad, fm for flat model, cm for conv model and cam for conv alternate model", default='cm')
  parser.add_argument('--p2', '-q', help="player2: r for random, a for Aliostad, fm for flat model, cm for conv model and cam for conv alternate model", default='a')
  parser.add_argument('--radius', '-r', help="radius of hexagon", type=int, default=4)
  parser.add_argument('--max_rounds', '-x', help="max rounds for a game", type=int)
  parser.add_argument('--intelligent_resource', '-i', help="intelligent resource selection for moves using PPO", type=bool, nargs='?', const=True)
  parser.add_argument('--no_ui', '-u', help="Not to serve API for UI", type=bool, nargs='?', const=True)
  parser.add_argument('--quantized_resource', '-z', help="intelligent resource selection for moves using quantized_resources", type=int, default=1)
  parser.add_argument('--training_model', '-t', help="training model: c for conv, ca for conv alternate and f for flat", default='c')

  return parser.parse_known_args(sys.argv[1:])


if __name__ == '__main__':

  def dummyDisplay(board):
    pass

  args = DotDict({
    'numIters': 20,
    'numEps': 10,
    'tempThreshold': 5,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 10,
    'cpuct': 1,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/', 'temp.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'default_quantization_proportion': 0.7
  })

  known, unknown = menu()
  args.update(known.__dict__)

  train = True
  test = False
  
  if len(sys.argv) > 1 and sys.argv[1] == 'test':
    train = False
    test = True
  g = HexagonGame(radius=args.radius, verbose=False,
                  resource_quantisation=args.quantized_resource, max_rounds=args.max_rounds)

  # conv model
  conv_model = HexagonModel(g)
  conv_model.load_checkpoint('models', 'conv_{}_{}.tar'.format(args.radius, g.resource_quantisation))

  # conv alt model
  conv_alt_model = HexagonAlternativeModel(g)
  conv_alt_model.load_checkpoint('models', 'conv_alt_{}_{}.tar'.format(args.radius, g.resource_quantisation))

  # flat model
  flat_model = HexagonFlatModel(g)
  flat_model.load_checkpoint('models', 'flat_{}_{}.tar'.format(args.radius, g.resource_quantisation))

  if not args.no_ui:
    hexagon_ui_api.run_in_background()

  training_models = {
    'c': conv_model,
    'ca': conv_alt_model,
    'f': flat_model
  }
  rm = ResourceModel(g)
  if os.path.exists('ppo_1_actor.h5f'):
    rm.model.load_weights('ppo_1_actor.h5f')
  if os.path.exists('ppo_1_critic.h5f'):
    rm.critic.load_weights('ppo_1_critic.h5f')
  g.intelligent_resource_actor = PPOAgent(1, rm.model, rm.critic,
           EpisodicMemory(experience_window_length=100000),
           (14,), name=PlayerNames.Player1, continuous=True,
           nb_steps_warmup=80, batch_size=200, training_epochs=64)
  if args.what == 'train':

    _player_name_mapper.register_player_name('alpha1', PlayerNames.Player1)
    _player_name_mapper.register_player_name('alpha2', PlayerNames.Player2)

    c = Coach(g, training_models[args.training_model], args)

    if args.load_model:
      print("Load trainExamples from file")
      c.loadTrainExamples()

    if args.intelligent_resource:
      g.intelligent_resource_players[PlayerIds.Player1] = True
      g.intelligent_resource_players[PlayerIds.Player2] = True
      g.intelligent_resource_actor.training = True
      g.intelligent_resource_actor.memories = {
        PlayerIds.Player1: EpisodicMemory(experience_window_length=100000),
        PlayerIds.Player2: EpisodicMemory(experience_window_length=100000)
      }

      def checkpoint():
        rm.model.save_weights('ppo_1_actor.h5f', overwrite=True)
        rm.critic.save_weights('ppo_1_critic.h5f', overwrite=True)

      c.checkpointing_event = checkpoint

    c.learn()
  
  if args.what == 'test':

    def get_player(v, id):
      """

      :type v: str
      :param id:
      :return:
      """
      if v == 'a':
        g.all_valid_moves_player = id
        name = 'aliostad' + str(id)
        return AliostadPlayer(g, name, am_i_second_player=id<0), name, False, False
      elif v.startswith('fm'):
        return CentaurPlayer(g, flat_model, args), 'flat_centaur' + str(id), v.endswith('i'), v.endswith('z')
      elif v.startswith('cm'):
        return CentaurPlayer(g, conv_model, args), 'conv_centaur' + str(id), v.endswith('i'), v.endswith('z')
      elif v.startswith('cam'):
        return CentaurPlayer(g, conv_alt_model, args), 'conv_alt_centaur' + str(id), v.endswith('i'), v.endswith('z')
      elif v == 'r':
        g.all_valid_moves_player = id
        return RandomPlayer(g, -1), 'random_player', False, False
      else:
        raise Exception("Invalid player: " + v)

    player1, player1_name, intelligent_resource_1, quantized_resource_1 = get_player(args.p1, 1)
    player2, player2_name, intelligent_resource_2, quantized_resource_2 = get_player(args.p2, -1)
    if intelligent_resource_1:
      player1_name += ' (i)'
      g.intelligent_resource_players[PlayerIds.Player1] = True
    if intelligent_resource_2:
      player2_name += ' (i)'
      g.intelligent_resource_players[PlayerIds.Player2] = True
    if quantized_resource_1:
      player1_name += ' (z)'
      del g.resource_quant_for_player[PlayerIds.Player1]
    if quantized_resource_2:
      player2_name += ' (z)'
      del g.resource_quant_for_player[PlayerIds.Player2]

    _player_name_mapper.register_player_name(player1_name, PlayerNames.Player1)
    _player_name_mapper.register_player_name(player2_name, PlayerNames.Player2)

    arena = Arena(player1.play, player2.play, g, display=dummyDisplay)
    print(arena.playGames(20, verbose=True))