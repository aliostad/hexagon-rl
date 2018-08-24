"""
An agent that acts on spaces that have discrete cells like many board games, such as chess.
In these spaces, the coordinates are discrete and environment is uniquely characterised by:

 - There is an inherent spatial neighbourhood-ness which can get lost if representations are flattened
 - On the other hand, a continuous representation of coordinates and then rounding can result in
   unacceptable results since due to noise output could point to a neighbouring location where
   some actions are meaningless.

This agent assigns scores to each of these discrete cells.

"""

from rl.core import Agent
from rl.memory import SequentialMemory
from short_memory import ShortMemory
import numpy as np
import keras

class DiscreteSpatial2DAgent(Agent):

  def _reset(self):
    self.not_committed_states = ShortMemory(self.reward_accumulation_steps)
    self.recent_rewards = ShortMemory(self.reward_accumulation_steps)

  def __init__(self, model, processor=None, memory=None, train_interval=100,
               batch_size=100, reward_accumulation_steps=10, random_exploration=0.1,
               random_variance=0.5, x_preparation=None, y_preparation=None,
               y_processing=None):
    """

    :param model:
    :param processor:
    :param memory:
    :param train_interval:
    :param batch_size:
    :param reward_cumulation_steps:
    :param random_exploration:
    :param random_variance:
    """
    Agent.__init__(self, processor=processor)

    self.model = model
    self.memory = SequentialMemory(10000) if memory is None else memory
    self.train_interval = train_interval
    self.batch_size = batch_size
    self.reward_accumulation_steps = reward_accumulation_steps
    self.random_exploration = random_exploration
    self.random_variance = random_variance
    self.x_preparation = x_preparation
    self.y_preparation = y_preparation
    self.y_processing = y_processing
    self._reset()

    # defaults
    self.current_action = None
    self.current_state = None
    self.since_last_batch = 0

  def _next_best_state(self, state):
    """
    expecting 2D array
    :type state: ndarray
    :return:
    """
    # find max
    oldMax = max(state.flatten())
    idx = np.argmax(flat, 0)
    x = idx % state.shape[1]
    y = idx / state.shape[1]
    state[y, x] /= 2
    newMax = max(state.flatten())
    state *= float(oldMax) / float(newMax)  # prevent decay
    return state

  def select_action(self, state):
    X = np.array(state)
    if self.x_preparation is not None:
      X = self.x_preparation(X)
    Y = self.model.predict(np.array([X]))[0]
    if self.y_processing is not None:
      Y = self.y_processing(Y)

    if np.random.uniform() < self.random_exploration:
      shape = Y.shape
      flat = Y.flatten()
      flat = np.array([i * np.random.uniform(0, self.random_variance) for i in flat])
      Y = np.reshape(flat, shape)
    return Y

  def forward(self, observation):
    self.current_state = observation
    self.current_action = self.select_action(observation)
    return self.current_action

  def backward(self, reward, terminal):
    self.recent_rewards.append(reward)
    bottom = self.not_committed_states.append((self.current_state, self.current_action))
    if bottom is not None:
      bottom_state, bottom_action = bottom
      n_positives = len(filter(lambda r: r > 0, self.recent_rewards))
      if n_positives > len(reward) - n_positives:  # if it has been generally more positive than negative or 0
        self.memory.append(bottom_state, bottom_action, sum(self.recent_rewards), False, self.training)

    if terminal:
      # finalise the states
      for idx, (state, action) in enumerate(self.not_committed_states):
        rewards = self.recent_rewards[idx:]
        n_positives = len(filter(lambda r: r > 0, rewards))
        if n_positives < len(reward) - n_positives:
          state = self._next_best_state(state)
        self.memory.append(state, action, sum(rewards), True if idx == self.reward_accumulation_steps - 1 else False,
                           self.training)

      self._reset()

    if self.training and self.step % self.train_interval == 0:
      self.train()

  def train(self):
    X = []
    Y = []
    experiences = self.memory.sample(self.batch_size)
    for e in experiences:
      X.append(e.state0)
      Y.append(e.action)
    X = np.array(X)
    Y = np.array(Y)
    if self.x_preparation is not None:
      X = self.x_preparation(X)
    if self.y_preparation is not None:
      Y = self.y_preparation(Y)
    self.model = keras.Model()
    self.model.train_on_batch(X, Y)

  def compile(self, optimizer, metrics=[]):
    self.model = keras.Model()
    self.model.compile(optimizer=optimizer)

  def load_weights(self, filepath):
    self.model.load_weights(filepath)

  def save_weights(self, filepath, overwrite=False):
    self.model.save_weights(filepath, overwrite)

  @property
  def layers(self):
    return self.model.layers


if __name__ == '__main__':
  pass

