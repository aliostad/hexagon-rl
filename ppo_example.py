from keras import Input, Model
from keras.layers import Dense
from keras.optimizers import Adam, SGD
import gym
from noisy_dense import NoisyDense

from ppo import *

LR = 3e-5
#ENV = 'Breakout-ram-v0'
ENV = 'LunarLander-v2'
EPISODES = 10000000

LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = 10

GAMMA = 0.99

BATCH_SIZE = 256
NUM_ACTIONS = 4
NUM_STATE = 8
ONLY_LAST_EPISODE=False

def build_actor():
  state_input = Input(shape=(NUM_STATE,))
  advantage = Input(shape=(1,))
  old_prediction = Input(shape=(NUM_ACTIONS,))

  x = Dense(256, activation='relu')(state_input)
  x = Dense(256, activation='relu')(x)

  # Prefer this to entropy penalty
  out_actions = Dense(NUM_ACTIONS, activation='softmax', name='output')(x)

  model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
  model.compile(optimizer=SGD(lr=LR),
                loss=[PPOAgent.proximal_policy_optimization_loss(
                  advantage=advantage,
                  old_prediction=old_prediction)])
  model.summary()

  return model


def build_critic():
  state_input = Input(shape=(NUM_STATE,))
  x = Dense(256, activation='relu')(state_input)
  x = Dense(256, activation='relu')(x)

  out_value = Dense(1)(x)

  model = Model(inputs=[state_input], outputs=[out_value])
  model.compile(optimizer=SGD(lr=LR), loss='mse')

  return model

if __name__ == '__main__':
  agent = PPOAgent(NUM_ACTIONS, build_actor(), build_critic(),
                   EpisodicMemory(100000, GAMMA, only_last_episode=ONLY_LAST_EPISODE),
                   observation_shape=(NUM_STATE, ), train_on_last_episode=ONLY_LAST_EPISODE,
                   train_interval=32, batch_size=BATCH_SIZE)
  env = gym.make(ENV)
  agent.fit(env, EPISODES, visualize=False, verbose=0)
