from keras.layers import Flatten, Conv2D
from keras.optimizers import Adam, Adagrad
from rl.agents import DQNAgent, CEMAgent
from rl.memory import SequentialMemory, EpisodeParameterMemory
from discrete_spatial_agent import DiscreteSpatial2DAgent

from centaur import *
from random import shuffle
from multi_agent import *
import sys
import hexagon_ui_api
import os
from square_grid import *
import numpy as np
from numpy.core.multiarray import *
import copy

# ______________________________________________________________________________________________________________________________
class EnvDef:
  centaur_name = 'centaur'
  game_name = '1'
  HASH_POOL = 10000
  NODE_FEATURE_COUNT = 5
  DECISION_ACTION_SPACE = 2
  SHORT_MEMORY_SIZE = 1
  MAX_ROUND = 2000
  CELL_FEATURE = 1
  MAX_GRID_LENGTH = 33
  SPATIAL_INPUT = (MAX_GRID_LENGTH, MAX_GRID_LENGTH)
  SPATIAL_OUTPUT = (MAX_GRID_LENGTH, MAX_GRID_LENGTH)

class AgentType:
  BoostDecision = 'BoostDecision'
  Attack = 'Attack'
  Boost = 'Boost'


# __________________________________________________________________________________________________________________________
class SuperCentaurPlayer(Aliostad):
  def __init__(self, name):
    Aliostad.__init__(self, name)
    self.reset_state()

  def reset_state(self):
    self.actions = {}
    self.current_move = None
    self.was_called = {}
    self.illegal_move_by_agents = {}

  #  just overriding for instrumentation purposes.
  def turnx(self, world):
    return Aliostad.turnx(self, world)

  def timeForBoost(self, world):
    """

    :type world: World
    :return:
    """
    #self.was_called[AgentType.BoostDecision] = True
    #return self.actions[AgentType.BoostDecision] == 1
    return np.random.uniform() < 0.23

  def move(self, playerView):
    mv = self.current_move
    return mv

  def getAttackFromCellId(self, world):
    self.was_called[AgentType.Attack] = True
    cellId = self.actions[AgentType.Attack]
    if cellId not in world.uberCells:
      self.illegal_move_by_agents[AgentType.Attack] = True
      print('illegal move: {}'.format(cellId))
      return None
    if not world.uberCells[cellId].canAttackOrExpand:
      self.illegal_move_by_agents[AgentType.Attack] = True
      print('illegal move: {}'.format(cellId))
      return None
    return cellId


# __________________________________________________________________________________________________________________________
class HierarchicalCentaurEnv(Env):
  def __init__(self):
    self.players = []
    self.game = None
    self._seed = 0
    self.centaur = None
    self.cell_count = 1
    self.resources = 100
    self.world = None
    self.leaderBoard = {}
    self.cellLeaderBoard = {}
    self.shortMemory = []

  def configure(self, *args, **kwargs):
    pass

  def seed(self, seed=None):
    self._seed = seed
    return [self._seed]

  def render(self, mode='human', close=False):
    pass

  def step(self, actions):
    for name in actions:
      self.centaur.actions[name] = actions[name]  # we can also deep copy

    self.centaur.current_move = self.centaur.movex(self.world)
    stats, isFinished = self.game.run_sync()
    info = self.game.board.get_cell_infos_for_player(EnvDef.centaur_name)
    reward = -1
    if len(info) == 0 or self.game.round_no > EnvDef.MAX_ROUND:
      isFinished = True  # it has no cells anymore
    else:
      reward = (len(info) - self.cell_count) if self.centaur.was_called else 0
      self.cell_count = len(info)
    if self.game.round_no % 100 == 0:
      print(self.cell_count)

    if isFinished:
      winner = stats[0]
      reward = 1000 if winner.playerName == EnvDef.centaur_name else -1000
      if winner.playerName in self.leaderBoard:
        self.leaderBoard[winner.playerName] += 1
      else:
        self.leaderBoard[winner.playerName] = 1
      for stat in stats:
        if stat.playerName not in self.cellLeaderBoard:
          self.cellLeaderBoard[stat.playerName] = 0
        self.cellLeaderBoard[stat.playerName] += stat.cellsOwned
        print('{} {} ({})'.format(stat.playerName, stat.cellsOwned, stat.totalResources))

      for name in self.leaderBoard:
        print(' - {}: {} ({})'.format(name, self.leaderBoard[name], self.cellLeaderBoard[name]))

    playerView = PlayerView(self.game.round_no, info)
    wrld = self.centaur.build_world(playerView.ownedCells)
    self.push_world(wrld)
    rewards = {name: -50 if name in self.centaur.illegal_move_by_agents else reward for name in self.centaur.was_called}
    self.centaur.reset_state()
    return wrld, rewards, isFinished, {}

  def push_world(self, world):
    """

    :type world: World
    :return:
    """
    self.world = world
    self.shortMemory.append(world)
    if len(self.shortMemory) > EnvDef.SHORT_MEMORY_SIZE:
      del self.shortMemory[0]

  def close(self):
    print('closing CentaurEnv')

  def reset(self):
    self.cell_count = 1
    self.resources = 100
    if self.game is not None:
      self.game.finish()

    self.shortMemory = []
    for i in range(0, EnvDef.SHORT_MEMORY_SIZE):
      self.shortMemory.append(World([]))

    self.centaur = SuperCentaurPlayer(EnvDef.centaur_name)
    self.players = [Aliostad('ali'), Aliostad('random80', 0.80), self.centaur, Aliostad('random30', 0.3),
                    Aliostad('random60', 0.6), Aliostad('random70', 0.7)]
    shuffle(self.players)
    self.game = Game(EnvDef.game_name, self.players, radius=11)
    hexagon_ui_api.games[EnvDef.game_name] = self.game
    self.game.start()
    playerView = PlayerView(self.game.round_no, self.game.board.get_cell_infos_for_player(EnvDef.centaur_name))
    wrld = self.centaur.build_world(playerView.ownedCells)
    self.push_world(wrld)
    return wrld

# ____________________________________________________________________________________________________________________________
class CentaurDecisionProcessor(Processor):

  @staticmethod
  def calculate_hash_index(cellName):
    # type: (str) -> int
    return int(abs(hash(cellName))) % EnvDef.HASH_POOL

  def buildInput(self, world):
    """
    Each input has 5 values:
      - first 4 items is 1-hot-vector:
        - No cell (we do not see)
        - My cell
        - Enemy cell
        - Not-captured cell
      - Resources

    :type world:
    :return:
    """
    inpt = [np.array([1, 0, 0, 0, 0]) for i in range(0, EnvDef.HASH_POOL)]

    for cell in world.cells.values():
      id = self.calculate_hash_index(str(cell.id))
      inpt[id] = np.array([0, 1, 0, 0, cell.resources])
      for n in cell.neighbours:
        id = int(abs(hash(n.id))) % EnvDef.HASH_POOL
        if n.isOwned is None:  # neutral
          inpt[id] = np.array(
            [0, 0, 0, 1, cell.resources])  # resources would be 0 but better just to use resources property
        elif n.isOwned is False:  # enemy
          inpt[id] = np.array([0, 0, 1, 0, cell.resources])
    inpt = np.array(inpt).flatten()
    return inpt

  def process_action(self, action):
    return action  # this is due to a BUG in keras-rl when it tries to calc mean

  def process_observation(self, observation):
    """

    :type observation: World
    :return:
    """
    return self.buildInput(observation)


# ______________________________________________________________________________________________________________________________
class CentaurAttackProcessor(Processor):
  def __init__(self, masking=True):
    self.masking = masking
    self.last_world = None

  def buildInput(self, world):
    """
    returns a MxN map of the world with hexagon grid transposed to square grid

    :type world: World
    :return:
    """
    self.last_world = world
    hector = np.zeros(EnvDef.SPATIAL_INPUT)
    for cid in world.worldmap:
      hid = GridCellId.fromHexCellId(cid)
      thid = hid.transpose(EnvDef.SPATIAL_INPUT[0] / 2, EnvDef.SPATIAL_INPUT[1] / 2)
      hector[thid.x][thid.y] = world.worldmap[cid]
    return hector

  def buildOutput(self, world):
    """
    returns a MxN map of the world with hexagon grid transposed to square grid

    :type world: World
    :return:
    """
    hector = np.zeros(EnvDef.SPATIAL_INPUT)
    for cid in world.uberCells:
      hid = GridCellId.fromHexCellId(cid)
      thid = hid.transpose(EnvDef.SPATIAL_INPUT[0] / 2, EnvDef.SPATIAL_INPUT[1] / 2)
      hector[thid.x][thid.y] = 0 if not world.uberCells[cid].canAttackOrExpand else world.uberCells[cid].attackPotential
    return hector

  def process_action(self, action):
    """

    :type action: ndarray
    :return:
    """
    flat = action.flatten()
    idx = np.argmax(flat, 0)
    y = idx % action.shape[1]
    x = idx / action.shape[1]
    if idx == 0:
      print('zero!')
    thid = GridCellId(x, y).transpose(-(EnvDef.SPATIAL_INPUT[0] / 2), -(EnvDef.SPATIAL_INPUT[1] / 2))
    return thid.to_cell_id()

  @staticmethod
  def add_noise(arr, noise_range=0.3):
    """

    :type arr: ndarray
    :type noise_range: float
    :return:
    """
    shp = arr.shape
    flat = arr.flatten()
    noisy = np.array([value * np.random.uniform(1. - noise_range, 1. + noise_range) for value in flat])
    return noisy.reshape(shp)

  def process_and_mask(self, Y):
    """

    :type Y: ndarray
    :return:
    """
    shibo = AttackModel.process_y(Y)
    real_shibo = shibo
    if self.masking and self.last_world is not None:
      if min(shibo.flat) == sum(shibo.flat):
        print('all zero output')
        return self.add_noise(self.buildOutput(self.last_world))
      mask = np.zeros(shibo.shape)
      for cid in self.last_world.uberCells:
        if self.last_world.uberCells[cid].canAttackOrExpand:
          hid = GridCellId.fromHexCellId(cid)
          thid = hid.transpose(EnvDef.SPATIAL_INPUT[0] / 2, EnvDef.SPATIAL_INPUT[1] / 2)
          mask[thid.x][thid.y] = 1
      minimum = min(shibo.flatten())
      if minimum < 0:  # rescale to zero
        shibo += -minimum
      real_shibo = shibo * mask
      if sum(real_shibo.flat) == 0:
        print('all zero output after masking')
        return self.add_noise(self.buildOutput(self.last_world))
    return real_shibo

  def process_observation(self, observation):
    """

    :type observation: World
    :return:
    """
    return self.buildInput(observation)


class CentaurBoostProcessor(CentaurAttackProcessor):

  def __init__(self, masking=True):
    CentaurAttackProcessor.__init__(self, masking)

  def process_and_mask(self, Y):
    """

    :type Y: ndarray
    :return:
    """
    shibo = AttackModel.process_y(Y)
    real_shibo = shibo
    if self.masking and self.last_world is not None:
      mask = np.zeros(Y.shape)
      for cid in self.last_world.uberCells:
        hid = GridCellId.fromHexCellId(cid)
        thid = hid.transpose(EnvDef.SPATIAL_INPUT[0] / 2, EnvDef.SPATIAL_INPUT[1] / 2)
        mask[thid.x][thid.y] = 1
      real_shibo = shibo * mask
      if max(real_shibo.flatten()) == 0:
        print('what??')
    return real_shibo
# ______________________________________________________________________________________________________________________________


class DecisionModel:
  def __init__(self, modelName=None):
    """

    :type theMethod: str
    """
    self.modelName = modelName if modelName is not None else 'Decision_model_params.h5f' + str(r.uniform(0, 10000))
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + (EnvDef.HASH_POOL * EnvDef.NODE_FEATURE_COUNT * EnvDef.SHORT_MEMORY_SIZE,)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(EnvDef.DECISION_ACTION_SPACE))
    model.add(Activation('softmax'))
    print(model.summary())
    model.compile(loss="categorical_crossentropy",
                  optimizer='adadelta', metrics=['accuracy'])
    self.model = model


# ______________________________________________________________________________________________________________________________


class AttackModel:
  def __init__(self, modelName=None):
    """

    :type theMethod: str
    """
    self.modelName = modelName if modelName is not None else 'Attack_model_params.h5f' + str(r.uniform(0, 10000))

    model = Sequential()
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=EnvDef.SPATIAL_INPUT + (1, )))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(4, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(1, (1, 1), padding='same', activation='relu'))

    if os.path.exists(self.modelName):
      model.load_weights(self.modelName)
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

    self.model = model

  @staticmethod
  def prepare_x(X, batch=False):
    """

    :type X: ndarray
    :type batch: bool
    :return:
    """
    shape = EnvDef.SPATIAL_INPUT + (1, )
    if batch:
      shape = (X.shape[0], ) + shape
    return np.reshape(X, shape)

  @staticmethod
  def prepare_y(Y, batch=False):
    """

    :type Y: ndarray
    :type batch: bool
    :return:
    """
    shape = EnvDef.SPATIAL_OUTPUT + (1, )
    if batch:
      shape = (Y.shape[0], ) + shape
    return np.reshape(Y, shape)

  @staticmethod
  def process_y(Y):
    """

    :type Y: ndarray
    :return:
    """
    return np.reshape(Y, EnvDef.SPATIAL_OUTPUT)

# ______________________________________________________________________________________________________________________________



if __name__ == '__main__':
  env = HierarchicalCentaurEnv()
  np.random.seed(42)
  env.seed(42)

  if len(sys.argv) > 2:
    method = sys.argv[2]

  prc = CentaurDecisionProcessor()
  dec_model = DecisionModel()
  attack_model = AttackModel('Attack_model_params.h5f')

  prc = MultiProcessor({AgentType.BoostDecision: prc, AgentType.Attack: CentaurAttackProcessor()})
  memory = EpisodeParameterMemory(limit=1000, window_length=1)
  decision_agent = CEMAgent(model=dec_model.model, nb_actions=EnvDef.DECISION_ACTION_SPACE, memory=memory,
                            batch_size=50, nb_steps_warmup=200, train_interval=50, elite_frac=0.05)

  decision_agent.compile()
  memory2 = EpisodeParameterMemory(limit=1000, window_length=1)
  attack_agent = DiscreteSpatial2DAgent(attack_model.model, x_preparation=AttackModel.prepare_x,
                                        y_preparation=AttackModel.prepare_y,
                                        y_processing=
                                        prc.inner_processors[AgentType.Attack].process_and_mask)

  agent = MultiAgent({AgentType.BoostDecision: decision_agent, AgentType.Attack: attack_agent}, processor=prc)

  hexagon_ui_api.run_in_background()
  if len(sys.argv) == 1:
    print('Usage: python centaur_ai_gym.py (train|test)')
  elif sys.argv[1] == 'train':
    agent.fit(env, nb_steps=300 * 1000, visualize=False, verbose=2, interim_filenames={AgentType.Attack: attack_model.modelName})
    agent.save_weights({AgentType.BoostDecision: dec_model.modelName, AgentType.Attack: attack_model.modelName}, overwrite=True)
  elif sys.argv[1] == 'test':
    agent.test(env, nb_episodes=100)
  else:
    print('argument not recognised: ' + sys.argv[1])
