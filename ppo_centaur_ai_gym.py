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
from ppo import PPOAgent
from episodic_memory import EpisodicMemory
from ai_gym import *

# ______________________________________________________________________________________________________________________________
class EnvDef:
  HASH_POOL = 10000
  NODE_FEATURE_COUNT = 5
  DECISION_ACTION_SPACE = 2
  MAX_ROUND = 2000
  CELL_FEATURE = 1
  MAX_GRID_LENGTH = 13
  RADIUS = (MAX_GRID_LENGTH/2) + 1
  SPATIAL_INPUT = (MAX_GRID_LENGTH, MAX_GRID_LENGTH)
  SPATIAL_OUTPUT = (MAX_GRID_LENGTH * MAX_GRID_LENGTH, )
  WARMUP = (MAX_GRID_LENGTH ** 2) * 5
  LR = 0.0001

# __________________________________________________________________________________________________________________________


class DecisionModel:
  def __init__(self, modelName=None):
    """

    :type theMethod: str
    """
    self.modelName = modelName if modelName is not None else 'Decision_model_params.h5f' + str(r.uniform(0, 10000))
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + (EnvDef.HASH_POOL * EnvDef.NODE_FEATURE_COUNT,)))
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

  def __init__(self, modelName='Attack_model_params.h5f'):
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

  args = menu()
  env = HierarchicalCentaurEnv(opponent_randomness=args.randomness,
                               centaur_boost_likelihood=args.centaur_boost_likelihood,
                               boosting_off=args.boostingoff, attack_off=args.attackoff)
  np.random.seed(42)
  env.seed(42)

  prc = CentaurDecisionProcessor()
  dec_model = DecisionModel()
  attack_model = AttackModel()

  prc = MultiProcessor({AgentType.BoostDecision: prc, AgentType.Attack: CentaurAttackProcessor(EnvDef.SPATIAL_INPUT, random_action=args.randomaction)})
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
                                   memory=memory2, nb_steps_warmup=EnvDef.WARMUP,
                                   masker=prc.inner_processors[AgentType.Attack])


  agent = MultiAgent({AgentType.BoostDecision: decision_agent, AgentType.Attack: attack_agent}, processor=prc, save_frequency=0.05)
  if args.model_name is not None:
    agent.inner_agents[AgentType.Attack].load_weights(args.model_name)

  if len(sys.argv) == 1:
    print('Usage: python centaur_ai_gym.py (train|test)')
  elif args.what == 'train':
    hexagon_ui_api.run_in_background()
    agent.fit(env, nb_steps=300 * 1000, visualize=False, verbose=2, interim_filenames={AgentType.Attack: attack_model.modelName})
    agent.save_weights({AgentType.BoostDecision: dec_model.modelName, AgentType.Attack: attack_model.modelName}, overwrite=True)
  elif args.what == 'test':
    hexagon_ui_api.run_in_background()
    agent.test(env, nb_episodes=args.testrounds)
  else:
    print('argument not recognised: ' + sys.argv[1])
