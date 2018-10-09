from keras import Input, Model, Sequential
from keras.layers import Flatten, Conv2D, Concatenate, Dense, Activation, concatenate
from keras.optimizers import Adam, SGD
from rl.agents import CEMAgent
from rl.memory import SequentialMemory, EpisodeParameterMemory

from hexagon_agent import *
from random import shuffle
from multi_agent import *
import sys
import hexagon_ui_api
import os
from square_grid import *
import numpy as np
import warnings
import argparse
from ppo import PPOAgent
from episodic_memory import EpisodicMemory
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
  MAX_GRID_LENGTH = 9
  SPATIAL_INPUT = (MAX_GRID_LENGTH, MAX_GRID_LENGTH)
  SPATIAL_OUTPUT = (MAX_GRID_LENGTH * MAX_GRID_LENGTH, )
  EPISODE_REWARD = 1000
  MOVE_REWARD_MULTIPLIER = 10
  DONT_OWN_MOVE_REWARD = -5
  CANT_ATTACK_MOVE_REWARD = -3
  LR = 0.0001

# __________________________________________________________________________________________________________________________

# __________________________________________________________________________________________________________________________

class AgentType:
  BoostDecision = 'BoostDecision'
  Attack = 'Attack'
  Boost = 'Boost'

# __________________________________________________________________________________________________________________________
class SuperCentaurPlayer(Aliostad):
  def __init__(self, name, boost_likelihood=0.23, boosting_off=False, attack_off=False):
    Aliostad.__init__(self, name)
    self.reset_state()
    self.boost_likelihood = boost_likelihood
    self.boosting_off = boosting_off
    self.attack_off = attack_off

  def reset_state(self):
    self.actions = {}
    self.current_move = None
    self.was_called = {}
    self.illegal_move_reward_by_agents = {}

  def timeForBoost(self, world):
    """

    :type world: World
    :return:
    """
    if self.boosting_off:
      return Aliostad.timeForBoost(self, world)

    #self.was_called[AgentType.BoostDecision] = True
    #return self.actions[AgentType.BoostDecision] == 1
    isTimeForBoost = np.random.uniform() < self.boost_likelihood
    self.boost_stats.append(isTimeForBoost)
    return isTimeForBoost

  def move(self, playerView):
    return self.current_move

  def getAttackFromCellId(self, world):
    if self.attack_off:
      return Aliostad.getAttackFromCellId(self, world)

    self.was_called[AgentType.Attack] = True
    cellId = self.actions[AgentType.Attack]
    if cellId not in world.uberCells:
      self.illegal_move_reward_by_agents[AgentType.Attack] = EnvDef.DONT_OWN_MOVE_REWARD
      print('{} - illegal move (dont own): {}'.format(world.round_no, cellId))
      return None
    if not world.uberCells[cellId].canAttackOrExpand:
      self.illegal_move_reward_by_agents[AgentType.Attack] = EnvDef.CANT_ATTACK_MOVE_REWARD
      print('{} - illegal move (cant attack): {}'.format(world.round_no, cellId))
      return None
    #print ('{} - legal!!'.format(world.round_no))
    return cellId

# __________________________________________________________________________________________________________________________
class HierarchicalCentaurEnv(Env):
  def __init__(self, centaur_boost_likelihood, opponent_randomness=None, boosting_off=False, attack_off=False):
    self.boosting_off = boosting_off
    self.attack_off = attack_off
    self.opponent_randomness = opponent_randomness
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
    self.centaur_boost_likelihood = centaur_boost_likelihood

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
    stats, isFinished , extraInfo = self.game.run_sync()
    info = self.game.board.get_cell_infos_for_player(EnvDef.centaur_name)
    reward = (len(info) - self.cell_count) * EnvDef.MOVE_REWARD_MULTIPLIER
    if len(info) == 0 or self.game.round_no > EnvDef.MAX_ROUND:
      isFinished = True  # it has no cells anymore
    else:
      self.cell_count = len(info)
    if self.game.round_no % 100 == 0:
      print(self.cell_count)
    if isFinished:
      winner = stats[0]
      reward = EnvDef.EPISODE_REWARD if winner.playerName == EnvDef.centaur_name else -EnvDef.EPISODE_REWARD
      if winner.playerName in self.leaderBoard:
        self.leaderBoard[winner.playerName] += 1
      else:
        self.leaderBoard[winner.playerName] = 1
      for stat in stats:
        if stat.playerName not in self.cellLeaderBoard:
          self.cellLeaderBoard[stat.playerName] = 0
        self.cellLeaderBoard[stat.playerName] += stat.cellsOwned
        print('{} {} ({}) - {}'.format(stat.playerName, stat.cellsOwned, stat.totalResources, str(extraInfo[stat.playerName])))

      for name in self.leaderBoard:
        print(' - {}: {} ({})'.format(name, self.leaderBoard[name], self.cellLeaderBoard[name]))

    playerView = PlayerView(self.game.round_no, info)
    wrld = self.centaur.build_world(playerView.ownedCells)
    self.push_world(wrld)
    rewards = {name: (reward + self.centaur.illegal_move_reward_by_agents[name]) if name in self.centaur.illegal_move_reward_by_agents else
      reward for name in self.centaur.was_called}
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
      self.shortMemory.append(World([], 0))

    self.centaur = SuperCentaurPlayer(EnvDef.centaur_name,
                      boost_likelihood=self.centaur_boost_likelihood,
                      attack_off=self.attack_off, boosting_off=self.boosting_off)
    self.players = [self.centaur, Aliostad('aliostad', randomBoostFactor=self.opponent_randomness)]
    shuffle(self.players)
    self.game = Game(EnvDef.game_name, self.players, radius=5)
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
  def __init__(self, masking=True, random_action=False):
    self.masking = masking
    self.last_world = None
    self.random_action = random_action

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

  def process_action_old(self, action):
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

  def process_action(self, action):
    """

    :type action: int
    :return:
    """
    idx = action
    y = idx % EnvDef.SPATIAL_INPUT[1]
    x = idx / EnvDef.SPATIAL_INPUT[1]
    thid = GridCellId(x, y).transpose(-(EnvDef.SPATIAL_INPUT[0] / 2), -(EnvDef.SPATIAL_INPUT[1] / 2))
    cid = thid.to_cell_id()
    return cid

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

  def mask(self, Y):
    """

    :type Y: ndarray
    :return:
    """
    assert len(Y.shape) == 1  # it is flat
    if self.last_world is None:
      warnings.warn("Last world is None. Could not mask.")
      return Y
    mask = self.buildOutput(self.last_world).flatten()
    assert mask.shape == Y.shape
    mn = np.min(Y)
    all_zeros = np.count_nonzero(Y) == 0
    plus = (-mn*2) if mn < 0 else 0
    for i in range(0, len(Y)):
      Y[i] += plus  # get rid of negative values
      if mask[i] == 0:
        Y[i] = 0
      elif self.random_action or all_zeros:  # in case of all_zero, choose random from mask
        Y[i] = np.random.uniform()
    if Y.sum() == 0:
      if mask.sum() == 0:
        mask[0] = 1.
      return mask
    else:
      Y /= Y.sum()  # normalise
      return Y

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
    return np.reshape(self.buildInput(observation), EnvDef.SPATIAL_INPUT + (1, ))

  def process_state_batch(self, batch):
    """

    :type batch: ndarray
    :return:
    """
    return np.reshape(batch, (batch.shape[0], ) + EnvDef.SPATIAL_INPUT + (1, ))


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
    self.modelName = modelName if modelName is not None else 'Attack_model_params.h5f' + str(r.uniform(0, 10000000))

    state_input = Input(shape=(EnvDef.SPATIAL_INPUT + (1, )))
    advantage = Input(shape=(1,))
    old_prediction = Input(EnvDef.SPATIAL_OUTPUT)

    conv_path = Conv2D(128, (5, 5), padding='same', activation='relu', name='INPUT_ATTACK')(state_input)
    conv_path = Conv2D(64, (3, 3), padding='same', activation='relu')(conv_path)
    conv_path = Conv2D(16, (3, 3), padding='same', activation='relu')(conv_path)
    conv_path = Conv2D(4, (3, 3), padding='same', activation='relu')(conv_path)
    conv_path = Conv2D(1, (3, 3), padding='same', activation='tanh')(conv_path)
    conv_path = Flatten()(conv_path)
    merged = concatenate([conv_path, advantage, old_prediction], axis=1)
    merged = Dense(EnvDef.SPATIAL_OUTPUT[0], activation='tanh')(merged)
    actor_output = Dense(EnvDef.SPATIAL_OUTPUT[0], activation='tanh')(merged)
    model = Model(inputs=[state_input, advantage, old_prediction], outputs=[actor_output])
    model.compile(optimizer=Adam(lr=EnvDef.LR),
                  loss=[PPOAgent.proximal_policy_optimization_loss(
                    advantage=advantage,
                    old_prediction=old_prediction)])
    '''
    [PPOAgent.proximal_policy_optimization_loss(
                    advantage=advantage,
                    old_prediction=old_prediction)]
    '''
    self.model = model
    critic_input = Input(shape=(EnvDef.SPATIAL_INPUT + (1, )))
    critic_path = Flatten()(critic_input)
    critic_path = Dense(256, activation='relu')(critic_path)
    critic_path = Dense(256, activation='relu')(critic_path)
    critic_out = Dense(1)(critic_path)
    critic = Model(inputs=[critic_input], outputs=[critic_out])
    critic.compile(optimizer=Adam(lr=EnvDef.LR), loss='mse')
    self.critic = critic


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

  parser = argparse.ArgumentParser()
  parser.add_argument('what', help="what to do", type=str)
  parser.add_argument('--model_name', '-m', help="model name", type=str)
  parser.add_argument('--randomness', '-r', help="randomness of aliostad", type=float)
  parser.add_argument('--randomaction', '-x', help="action completely random but valid", type=bool, nargs='?', const=True)
  parser.add_argument('--boostingoff', '-y', help="don't use boosting method of centaur",
                      type=bool, nargs='?', const=True, default=True)
  parser.add_argument('--attackoff', '-z', help="dont use attack method of centaur",
                      type=bool, nargs='?', const=True, default=False)
  parser.add_argument('--testrounds', '-t', help="number of epochs when testing", type=int, default=100)
  parser.add_argument('--centaur_boost_likelihood', '-b', help="likelihood of random boost for centaur", type=float, default=0.23)

  randomness = None
  attack_model_name = 'Attack_model_params.h5f'

  args = parser.parse_args(sys.argv[1:])
  if args.model_name is not None:
    attack_model_name = args.model_name

  if args.randomness is not None:
    randomness = args.randomness

  env = HierarchicalCentaurEnv(opponent_randomness=randomness,
                               centaur_boost_likelihood=args.centaur_boost_likelihood,
                               boosting_off=args.boostingoff, attack_off=args.attackoff)
  np.random.seed(42)
  env.seed(42)

  prc = CentaurDecisionProcessor()
  dec_model = DecisionModel()
  attack_model = AttackModel(attack_model_name)

  prc = MultiProcessor({AgentType.BoostDecision: prc, AgentType.Attack: CentaurAttackProcessor(random_action=args.randomaction)})
  memory = EpisodeParameterMemory(limit=1000, window_length=1)
  decision_agent = CEMAgent(model=dec_model.model, nb_actions=EnvDef.DECISION_ACTION_SPACE, memory=memory,
                            batch_size=50, nb_steps_warmup=200, train_interval=50, elite_frac=0.05)

  decision_agent.compile()
  memory2 = EpisodicMemory(experience_window_length=100000)
  attack_agent = PPOAgent(nb_actions=EnvDef.SPATIAL_OUTPUT[0],
                          observation_shape=EnvDef.SPATIAL_INPUT + (1, ),
                                   actor=attack_model.model,
                                   processor=prc.inner_processors[AgentType.Attack],
                                   critic=attack_model.critic,
                                   memory=memory2, nb_steps_warmup=400,
                                   masker=prc.inner_processors[AgentType.Attack])


  agent = MultiAgent({AgentType.BoostDecision: decision_agent, AgentType.Attack: attack_agent}, processor=prc, save_frequency=0.05)
  if args.model_name is not None:
    agent.inner_agents[AgentType.Attack].load_weights(attack_model_name)

  hexagon_ui_api.run_in_background()
  if len(sys.argv) == 1:
    print('Usage: python centaur_ai_gym.py (train|test)')
  elif args.what == 'train':
    agent.fit(env, nb_steps=300 * 1000, visualize=False, verbose=2, interim_filenames={AgentType.Attack: attack_model.modelName})
    agent.save_weights({AgentType.BoostDecision: dec_model.modelName, AgentType.Attack: attack_model.modelName}, overwrite=True)
  elif args.what == 'test':
    agent.test(env, nb_episodes=args.testrounds)
  else:
    print('argument not recognised: ' + sys.argv[1])
