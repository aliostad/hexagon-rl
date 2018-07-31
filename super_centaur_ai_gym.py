from keras.layers import Flatten
from keras.optimizers import Adam
from rl.agents import DQNAgent, CEMAgent
from rl.core import Env, Processor
from rl.memory import SequentialMemory, EpisodeParameterMemory
from rl.policy import BoltzmannQPolicy

from centaur import *
from random import shuffle
from multi_agent import *
import os
import sys
import hexagon_ui_api

# ______________________________________________________________________________________________________________________________
class EnvDef:
  centaur_name = 'centaur'
  game_name = '1'
  HASH_POOL = 10000
  NODE_FEATURE_COUNT = 5
  ACTION_SPACE = 2
  SHORT_MEMORY_SIZE = 4
  MAX_ROUND = 2000

class AgentType:
  BoostDecision = 'BoostDecision'
  Attack = 'Attack'
  Boost = 'Boost'


# __________________________________________________________________________________________________________________________
class SuperCentaurPlayer(Aliostad):
  def __init__(self, name):
    Aliostad.__init__(self, name)
    self._reset_state()

  def _reset_state(self):
    self.actions = {}
    self.current_move = None
    self.was_called = {}

  def timeForBoost(self, world):
    """

    :type world: World
    :return:
    """
    self.was_called[AgentType.BoostDecision] = True
    return self.actions[AgentType.BoostDecision] == 1

  def move(self, playerView):
    mv = self.current_move
    self._reset_state()
    return mv
'''
  def getAttack(self, world):
    self.was_called[AgentType.Attack] = True
    return self.actions[AgentType.Attack]
'''

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
        print('{} {} ({})'.format(stat.playerName, stat.cellsOwned, stat.totalResources))

      for name in self.leaderBoard:
        print(' - {}: {}'.format(name, self.leaderBoard[name]))

    return PlayerView(self.game.round_no, info), {name: reward for name in actions}, isFinished, {}

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
    self.players = [Aliostad('ali'), Aliostad('random80', 0.80), self.centaur, Aliostad('random50', 0.5), Aliostad('random60', 0.6), Aliostad('random70', 0.7)]
    shuffle(self.players)
    self.game = Game(EnvDef.game_name, self.players, radius=11)
    hexagon_ui_api.games[EnvDef.game_name] = self.game
    self.game.start()
    return PlayerView(self.game.round_no, self.game.board.get_cell_infos_for_player(EnvDef.centaur_name))

# ____________________________________________________________________________________________________________________________
class CentaurDecisionProcessor(Processor):
  def __init__(self, envi):
    """

    :type env: CentaurEnv
    """
    self.env = envi

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
      id = int(abs(hash(cell.id))) % EnvDef.HASH_POOL
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

    :type observation: PlayerView
    :return:
    """
    self.env.push_world(self.env.centaur.build_world(observation.ownedCells))
    inpt = np.array([self.buildInput(w) for w in self.env.shortMemory])

    return inpt.flatten()



class DecisionModel:
  def __init__(self, theMethod):
    """

    :type theMethod: str
    """
    self.modelName = '{}_params.h5f'.format(theMethod) + str(r.uniform(0, 10000))
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + (EnvDef.HASH_POOL * EnvDef.NODE_FEATURE_COUNT * EnvDef.SHORT_MEMORY_SIZE,)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(EnvDef.ACTION_SPACE))
    model.add(Activation('softmax'))
    print(model.summary())
    model.compile(loss="categorical_crossentropy",
                  optimizer='adadelta', metrics=['accuracy'])
    self.model = model


# ______________________________________________________________________________________________________________________________


if __name__ == '__main__':
  env = HierarchicalCentaurEnv()
  np.random.seed(42)
  env.seed(42)

  method = 'CEM'
  if len(sys.argv) > 2:
    method = sys.argv[2]
  agent = None

  prc = CentaurDecisionProcessor(env)
  dec_model = DecisionModel(method)

  if method == 'DQN':
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    agent = DQNAgent(model=dec_model.model, nb_actions=EnvDef.ACTION_SPACE, memory=memory, nb_steps_warmup=10,
                   enable_dueling_network=True, dueling_type='avg',
                   target_model_update=0.01, policy=policy, processor=prc)

    agent.compile(Adam(lr=0.001), metrics=['mae'])
  elif method == 'CEM':
    memory = EpisodeParameterMemory(limit=1000, window_length=1)
    agent = CEMAgent(model=dec_model.model, nb_actions=EnvDef.ACTION_SPACE, memory=memory,
             batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05, processor=prc)
    agent.compile()
  elif method == 'SHUBBA':
    prc = MultiProcessor({AgentType.BoostDecision: prc})
    memory = EpisodeParameterMemory(limit=1000, window_length=1)
    agent = CEMAgent(model=dec_model.model, nb_actions=EnvDef.ACTION_SPACE, memory=memory,
             batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
    agent.compile()
    agent = MultiAgent({AgentType.BoostDecision: agent}, processor=prc)

  #hexagon_ui_api.run_in_background()
  if len(sys.argv) == 1:
    print('Usage: python centaur_ai_gym.py (train|test)')
  elif sys.argv[1] == 'train':
    agent.fit(env, nb_steps=3*100, visualize=False, verbose=2)
    agent.save_weights({AgentType.BoostDecision: dec_model.modelName}, overwrite=True)
  elif sys.argv[1] == 'test':
    agent.test(env, nb_episodes=100)
  else:
    print('argument not recognised: ' + sys.argv[1])
