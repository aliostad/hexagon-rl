from keras.layers import Flatten
from keras.models import load_model
from rl.agents import CEMAgent
from rl.core import Env, Processor
from rl.memory import EpisodeParameterMemory

from hexagon_gaming import *
from hexagon import *
from hexagon_agent import *
from centaur import *
import os
import sys

# _____________________________________________________________________________________________________________________________________
class EnvDef:
  centaur_name = 'centaur'
  game_name = '1'
  HASH_POOL = 10000
  NODE_FEATURE_COUNT = 5
  ACTION_SPACE = 2


# _____________________________________________________________________________________________________________________________________
class CentaurPlayer(Aliostad):
  def __init__(self, name):
    Aliostad.__init__(self, name)
    self.is_time_for_boost = None
    self.current_move = None
    self.was_called = False

  def timeForBoost(self, world):
    """

    :type world: World
    :return:
    """
    self.was_called = True
    return self.is_time_for_boost

  def move(self, playerView):
    self.was_called = False
    return self.current_move


# _____________________________________________________________________________________________________________________________________
class CentaurEnv(Env):
  def __init__(self):
    self.players = []
    self.game = None
    self._seed = 0
    self.centaur = None
    self.cell_count = 1
    self.resources = 100
    self.world = None

  def configure(self, *args, **kwargs):
    pass

  def seed(self, seed=None):
    self._seed = seed
    return [self._seed]

  def render(self, mode='human', close=False):
    pass

  def step(self, action):
    self.centaur.is_time_for_boost = action == 1
    action = self.centaur.movex(self.world)

    self.centaur.current_move = action
    stats, isFinished = self.game.run_sync()
    info = self.game.board.get_cell_infos_for_player(EnvDef.centaur_name)
    reward = -1
    if len(info) == 0:
      isFinished = True  # it has no cells anymore
    else:
      reward = (len(info) - self.cell_count) if self.centaur.was_called else 0
      self.cell_count = len(info)
    if self.game.round_no % 100 == 0:
      print(self.cell_count)

    if isFinished:
      for stat in stats:
        print('{} {} ({})'.format(stat.playerName, stat.cellsOwned, stat.totalResources))
    return PlayerView(self.game.round_no, info), reward, isFinished, {}

  def close(self):
    print('closing CentaurEnv')

  def reset(self):
    self.cell_count = 1
    self.resources = 100
    if self.game is not None:
      self.game.finish()

    self.centaur = CentaurPlayer(EnvDef.centaur_name)
    self.players = [self.centaur, Aliostad('ali'), Aliostad('random3', 0.3), Aliostad('random5', 0.5)]
    self.game = Game(EnvDef.game_name, self.players, radius=9)
    self.game.start()
    return PlayerView(self.game.round_no, self.game.board.get_cell_infos_for_player(EnvDef.centaur_name))


# _____________________________________________________________________________________________________________________________________
class CentaurProcessor(Processor):
  def __init__(self, envi):
    """

    :type env: CentaurEnv
    """
    self.env = envi
    self.world = None

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
    self.env.world = self.env.centaur.build_world(observation.ownedCells)
    return self.buildInput(self.env.world)

# _____________________________________________________________________________________________________________________________________


if __name__ == '__main__':
  env = CentaurEnv()
  np.random.seed(42)
  env.seed(42)

  memory = EpisodeParameterMemory(limit=1000, window_length=1)

  modelName = 'cem_{}_params.h5f'.format('sisi')

  model = Sequential()
  model.add(Flatten(input_shape=(1, ) + (EnvDef.HASH_POOL * EnvDef.NODE_FEATURE_COUNT, )))
  model.add(Dense(48, activation="relu"))
  model.add(Dense(24, activation="relu"))
  model.add(Dense(EnvDef.ACTION_SPACE))
  model.add(Activation('softmax'))
  model.compile(loss="categorical_crossentropy",
                optimizer='adadelta', metrics=['accuracy'])

  cem = CEMAgent(model=model, nb_actions=EnvDef.ACTION_SPACE, memory=memory,
                 batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05, processor=CentaurProcessor(env))
  cem.compile()
  if os.path.exists(modelName):
    cem.load_weights(modelName)

  if len(sys.argv) == 1:
    print('Usage: python centaur_ai_gym.py (train|test)')
  elif sys.argv[1] == 'train':
    cem.fit(env, nb_steps=100000, visualize=False, verbose=2)
    cem.save_weights(modelName + str(r.uniform(0,10000)), overwrite=True)
  elif sys.argv[1] == 'test':
    cem.test(env, nb_episodes=1)
  else:
    print('argument not recognised: ' + sys.argv[1])