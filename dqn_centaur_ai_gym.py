from keras.layers import Flatten, Conv2D, Dense, Activation
from keras.optimizers import Adam
from keras import Sequential
from rl.agents import DQNAgent, CEMAgent
from rl.memory import SequentialMemory, EpisodeParameterMemory
from rl.policy import EpsGreedyQPolicy


from hexagon_agent import *
from random import shuffle
from multi_agent import *
import sys
import hexagon_ui_api
import os
import numpy as np
from ai_gym import *

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
  MAX_GRID_LENGTH = 5
  SPATIAL_INPUT = (MAX_GRID_LENGTH, MAX_GRID_LENGTH)
  SPATIAL_OUTPUT = (MAX_GRID_LENGTH * MAX_GRID_LENGTH, )
  EPISODE_REWARD = 1000
  MOVE_REWARD_MULTIPLIER = 10
  DONT_OWN_MOVE_REWARD = -5
  CANT_ATTACK_MOVE_REWARD = -3
  GAME_VERBOSE = False
  RADIUS = 3
# __________________________________________________________________________________________________________________________

class NoneZeroEpsGreedyQPolicy(EpsGreedyQPolicy):
  """Implement the epsilon greedy policy

  Eps Greedy policy either:

  - takes a random action with probability epsilon from Non-Zero Q-values
  - takes current best action with prob (1 - epsilon)
  """

  def __init__(self, eps=.1):
    super(EpsGreedyQPolicy, self).__init__()
    self.eps = eps

  def select_action(self, q_values):
    """Return the selected action

    # Arguments
        q_values (np.ndarray): List of the estimations of Q for each action

    # Returns
        Selection action
    """
    assert q_values.ndim == 1
    nb_actions = q_values.shape[0]
    if np.random.uniform() < self.eps:
      copy_q_values = np.copy(q_values)
      idx = np.argmax(q_values)
      copy_q_values[idx] = 0
      for i in range(0, nb_actions):
        val = copy_q_values[i]
        copy_q_values[i] = -1e8 if val == 0 else val * np.random.uniform()
      action = np.argmax(copy_q_values)
    else:
      action = np.argmax(q_values)
    return action

  def get_config(self):
    """Return configurations of EpsGreedyPolicy

    # Returns
        Dict of config
    """
    config = super(EpsGreedyQPolicy, self).get_config()
    config['eps'] = self.eps
    return config


# __________________________________________________________________________________________________________________________

class MaskableDQNAgent(DQNAgent):

  def __init__(self, model, policy=None, test_policy=None, enable_double_dqn=True, enable_dueling_network=False,
               dueling_type='avg', mask_processor=None, *args, **kwargs):
    DQNAgent.__init__(self, model, policy=policy, test_policy=test_policy,
                      enable_double_dqn=enable_double_dqn, enable_dueling_network=enable_dueling_network,
                      dueling_type=dueling_type, *args, **kwargs)
    self.mask_processor = mask_processor

  def forward(self, observation):
    # Select an action.
    state = self.memory.get_recent_state(observation)
    q_values = self.compute_q_values(state)
    if self.mask_processor is not None:
      q_values = self.mask_processor.mask(q_values)
    if self.training:
      action = self.policy.select_action(q_values=q_values)
    else:
      action = self.test_policy.select_action(q_values=q_values)

    # Book-keeping.
    self.recent_observation = observation
    self.recent_action = action

    return action

# __________________________________________________________________________________________________________________________



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

class SimplestAttackModel:
  def __init__(self, modelName=None):
    """

    :type theMethod: str
    """
    self.modelName = modelName if modelName is not None else 'Attack_model_params.h5f' + str(r.uniform(0, 10000000))

    model = Sequential()
    model.add(Flatten(
              input_shape=EnvDef.SPATIAL_INPUT + (1, ), name='INPUT_ATTACK'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(EnvDef.SPATIAL_OUTPUT[0], activation='softmax'))

    self.model = model


class SimpleAttackModel:
  def __init__(self, modelName=None):
    """

    :type theMethod: str
    """
    self.modelName = modelName if modelName is not None else 'Attack_model_params.h5f' + str(r.uniform(0, 10000000))

    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
              input_shape=EnvDef.SPATIAL_INPUT + (1, ), name='INPUT_ATTACK'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(1, (1, 1), padding='same', activation='tanh'))
    model.add(Flatten())
    model.add(Dense(EnvDef.SPATIAL_OUTPUT[0], activation='tanh'))

    self.model = model


class AttackModel:
  def __init__(self, modelName=None):
    """

    :type theMethod: str
    """
    self.modelName = modelName if modelName is not None else 'Attack_model_params.h5f' + str(r.uniform(0, 10000000))

    model = Sequential()
    model.add(Conv2D(128, (5, 5), padding='same', activation='relu',
              input_shape=EnvDef.SPATIAL_INPUT + (1, ), name='INPUT_ATTACK'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(4, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(1, (3, 3), padding='same', activation='tanh'))
    model.add(Flatten())
    model.add(Dense(EnvDef.SPATIAL_OUTPUT[0], activation='tanh'))

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

  args = menu()
  env = HierarchicalCentaurEnv(opponent_randomness=args.randomness,
                               centaur_boost_likelihood=args.centaur_boost_likelihood,
                               boosting_off=args.boostingoff, attack_off=args.attackoff,
                               game_verbose=EnvDef.GAME_VERBOSE, radius=EnvDef.RADIUS,
                               move_shuffle=args.moveshuffle, move_handicap=args.handicap)
  np.random.seed(42)
  env.seed(42)

  prc = CentaurDecisionProcessor()
  dec_model = DecisionModel()
  attack_model = SimpleAttackModel('Attack_model_params.h5f')

  prc = MultiProcessor({AgentType.BoostDecision: prc, AgentType.Attack: CentaurAttackProcessor(EnvDef.SPATIAL_INPUT, random_action=args.randomaction)})
  memory = EpisodeParameterMemory(limit=1000, window_length=1)
  decision_agent = CEMAgent(model=dec_model.model, nb_actions=EnvDef.DECISION_ACTION_SPACE, memory=memory,
                            batch_size=50, nb_steps_warmup=200, train_interval=50, elite_frac=0.05)

  decision_agent.compile()
  memory2 = SequentialMemory(limit=100000, window_length=1)
  policy = NoneZeroEpsGreedyQPolicy()
  attack_agent = MaskableDQNAgent(attack_model.model,
                          policy=policy, batch_size=16,
                          processor=prc.inner_processors[AgentType.Attack],
                          nb_actions=EnvDef.SPATIAL_OUTPUT[0],
                          memory=memory2, nb_steps_warmup=500,
                          enable_dueling_network=True,
                          mask_processor=prc.inner_processors[AgentType.Attack] if args.usemasking else None)


  agent = MultiAgent({AgentType.BoostDecision: decision_agent, AgentType.Attack: attack_agent}, processor=prc, save_frequency=0.05)
  agent.inner_agents[AgentType.Attack].compile(Adam(lr=0.001), metrics=['mean_squared_logarithmic_error'])

  if args.model_name is not None:
    agent.inner_agents[AgentType.Attack].load_weights(args.model_name)

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
