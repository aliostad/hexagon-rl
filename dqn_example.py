from keras import Input, Model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam, SGD
import gym
from noisy_dense import NoisyDense

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from episodic_memory import EpisodicMemory

LR = 3e-5

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
  state_input = Input(shape=(1,) + (NUM_STATE,))
  x = Flatten()(state_input)
  x = Dense(256, activation='relu')(x)
  x = Dense(256, activation='relu')(x)

  out_actions = Dense(NUM_ACTIONS, activation='softmax', name='output')(x)

  model = Model(inputs=[state_input], outputs=[out_actions])
  model.summary()

  return model



if __name__ == '__main__':
  agent = DQNAgent(build_actor(), nb_actions=NUM_ACTIONS,
                   memory=SequentialMemory(limit=50000, window_length=1),
                   train_interval=32, batch_size=BATCH_SIZE, enable_dueling_network=True)
  agent.compile(Adam(lr=1e-3), metrics=['mae'])
  env = gym.make(ENV)
  agent.fit(env, EPISODES, visualize=True, verbose=2)
