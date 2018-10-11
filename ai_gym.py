import argparse
import sys

from hexagon_agent import *
from random import shuffle
from multi_agent import *
from square_grid import *
import hexagon_ui_api

#_________________________________________________________________________________________________________________________

class AgentType:
  BoostDecision = 'BoostDecision'
  Attack = 'Attack'
  Boost = 'Boost'

#_________________________________________________________________________________________________________________________

class SuperCentaurPlayer(Aliostad):
  def __init__(self, name, boost_likelihood=0.23, boosting_off=False, attack_off=False,
               dont_own_move_reward=-5, cant_attack_move_reward=-3):
    Aliostad.__init__(self, name)
    self.reset_state()
    self.boost_likelihood = boost_likelihood
    self.boosting_off = boosting_off
    self.attack_off = attack_off
    self.dont_own_move_reward = dont_own_move_reward
    self.cant_attack_move_reward = cant_attack_move_reward

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
      self.illegal_move_reward_by_agents[AgentType.Attack] = self.dont_own_move_reward
      print('{} - illegal move (dont own): {}'.format(world.round_no, cellId))
      return None
    if not world.uberCells[cellId].canAttackOrExpand:
      self.illegal_move_reward_by_agents[AgentType.Attack] = self.cant_attack_move_reward
      print('{} - illegal move (cant attack): {}'.format(world.round_no, cellId))
      return None
    #print ('{} - legal!!'.format(world.round_no))
    return cellId

# __________________________________________________________________________________________________________________________
class HierarchicalCentaurEnv(Env):
  def __init__(self, centaur_boost_likelihood, opponent_randomness=None,
               boosting_off=False, attack_off=False, centaur_name='centaur',
               game_name='1', move_reward_multiplier=10, max_rounds=2000,
               episode_reward=1000, radius=5):
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
    self.centaur_name = centaur_name
    self.game_name = game_name
    self.move_reward_multiplier = move_reward_multiplier
    self.max_rounds = max_rounds
    self.episode_reward = episode_reward
    self.radius = radius

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
    info = self.game.board.get_cell_infos_for_player(self.centaur_name)
    reward = (len(info) - self.cell_count) * self.move_reward_multiplier
    if len(info) == 0 or self.game.round_no > self.max_rounds:
      isFinished = True  # it has no cells anymore
    else:
      self.cell_count = len(info)
    if self.game.round_no % 100 == 0:
      print(self.cell_count)
    if isFinished:
      winner = stats[0]
      reward = self.episode_reward if winner.playerName == self.centaur_name else -self.episode_reward
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
    if len(self.shortMemory) > 1:
      del self.shortMemory[0]

  def close(self):
    print('closing CentaurEnv')

  def reset(self):
    self.cell_count = 1
    self.resources = 100
    if self.game is not None:
      self.game.finish()

    self.shortMemory = []
    for i in range(0, 1):
      self.shortMemory.append(World([], 0))

    self.centaur = SuperCentaurPlayer(self.centaur_name,
                      boost_likelihood=self.centaur_boost_likelihood,
                      attack_off=self.attack_off, boosting_off=self.boosting_off)
    self.players = [self.centaur, Aliostad('aliostad', randomBoostFactor=self.opponent_randomness)]
    shuffle(self.players)
    self.game = Game(self.game_name, self.players, radius=self.radius)
    hexagon_ui_api.games[self.game_name] = self.game
    self.game.start()
    playerView = PlayerView(self.game.round_no, self.game.board.get_cell_infos_for_player(self.centaur_name))
    wrld = self.centaur.build_world(playerView.ownedCells)
    self.push_world(wrld)
    return wrld


# __________________________________________________________________________________________________________________________
class CentaurAttackProcessor(Processor):
  def __init__(self, spatial_input, masking=True, random_action=False):
    self.masking = masking
    self.last_world = None
    self.random_action = random_action
    self.spatial_input = spatial_input

  def buildInput(self, world):
    """
    returns a MxN map of the world with hexagon grid transposed to square grid

    :type world: World
    :return:
    """
    self.last_world = world
    hector = np.zeros(self.spatial_input)
    for cid in world.worldmap:
      hid = GridCellId.fromHexCellId(cid)
      thid = hid.transpose(self.spatial_input[0] / 2, self.spatial_input[1] / 2)
      hector[thid.x][thid.y] = world.worldmap[cid]
    return hector

  def buildOutput(self, world):
    """
    returns a MxN map of the world with hexagon grid transposed to square grid

    :type world: World
    :return:
    """
    hector = np.zeros(self.spatial_input)
    for cid in world.uberCells:
      hid = GridCellId.fromHexCellId(cid)
      thid = hid.transpose(self.spatial_input[0] / 2, self.spatial_input[1] / 2)
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
    thid = GridCellId(x, y).transpose(-(self.spatial_input[0] / 2), -(self.spatial_input[1] / 2))
    return thid.to_cell_id()

  def process_action(self, action):
    """

    :type action: int
    :return:
    """
    idx = action
    y = idx % self.spatial_input[1]
    x = idx / self.spatial_input[1]
    thid = GridCellId(x, y).transpose(-(self.spatial_input[0] / 2), -(self.spatial_input[1] / 2))
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
          thid = hid.transpose(self.spatial_input[0] / 2, self.spatial_input[1] / 2)
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
    return np.reshape(self.buildInput(observation), self.spatial_input + (1, ))

  def process_state_batch(self, batch):
    """

    :type batch: ndarray
    :return:
    """
    return np.reshape(batch, (batch.shape[0], ) + self.spatial_input + (1, ))

# ____________________________________________________________________________________________________________________________
class CentaurDecisionProcessor(Processor):

  def __init__(self, hash_pool=10000):
    self.hash_pool = hash_pool

  def calculate_hash_index(self, cellName):
    # type: (str) -> int
    return int(abs(hash(cellName))) % self.hash_pool

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
    inpt = [np.array([1, 0, 0, 0, 0]) for i in range(0, self.hash_pool)]

    for cell in world.cells.values():
      id = self.calculate_hash_index(str(cell.id))
      inpt[id] = np.array([0, 1, 0, 0, cell.resources])
      for n in cell.neighbours:
        id = int(abs(hash(n.id))) % self.hash_pool
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

# __________________________________________________________________________________________________________________________
class CentaurBoostProcessor(CentaurAttackProcessor):

  def __init__(self, spatial_input, masking=True):
    CentaurAttackProcessor.__init__(self, spatial_input, masking)

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
        thid = hid.transpose(self.spatial_input[0] / 2, self.spatial_input[1] / 2)
        mask[thid.x][thid.y] = 1
      real_shibo = shibo * mask
      if max(real_shibo.flatten()) == 0:
        print('what??')
    return real_shibo

def menu():
  parser = argparse.ArgumentParser()
  parser.add_argument('what', help="what to do", type=str)
  parser.add_argument('--model_name', '-m', help="attack model name to load", type=str)
  parser.add_argument('--randomness', '-r', help="randomness of aliostad", type=float)
  parser.add_argument('--randomaction', '-x', help="action completely random but valid", type=bool, nargs='?', const=True)
  parser.add_argument('--boostingoff', '-y', help="don't use boosting method of centaur",
                      type=bool, nargs='?', const=True, default=True)
  parser.add_argument('--attackoff', '-z', help="dont use attack method of centaur",
                      type=bool, nargs='?', const=True, default=False)
  parser.add_argument('--testrounds', '-t', help="number of epochs when testing", type=int, default=100)
  parser.add_argument('--centaur_boost_likelihood', '-b', help="likelihood of random boost for centaur", type=float, default=0.23)
  return parser.parse_args(sys.argv[1:])

