from rl.memory import Memory
from collections import deque
import numpy as np

class EpisodeStep:
  def __init__(self, observation, action, reward, terminal, pred_action=None):
    self.observation = observation
    self.action = action
    self.reward = reward
    self.terminal = terminal
    self.discounted_reward = reward
    self.pred_action = pred_action

class Episode:
  def __init__(self, gamma):
    self.steps = []
    self.total_reward = 0.
    self.is_closed = False
    self.gamma = gamma

  def append(self, step, training=True):
    """

    :type step: EpisodeStep
    :type training: bool
    :return:
    """
    assert self.is_closed == False, 'Cannot append to a closed episode'
    self.total_reward += step.reward
    self.steps.append(step)
    if step.terminal:
      self.is_closed = True
    n = len(self.steps)
    for i, s in enumerate(self.steps):
      s.discounted_reward += step.reward * (self.gamma ** (n - i))


class EpisodicMemory(Memory):
  def __init__(self, experience_window_length, reward_decay_gamma=0.99, **kwargs):
    super(EpisodicMemory, self).__init__(window_length=experience_window_length, **kwargs)
    self.current_episode = Episode(reward_decay_gamma)
    self.gamma = reward_decay_gamma
    self.steps = deque(maxlen=experience_window_length)

  def sample(self, batch_size, batch_idxs=None):
    """

    :param batch_size: int, size of the batch
    :param batch_idxs: array, indices of !!episodes!! to choose from
    :return:
    """
    assert len(self.steps) > 0, 'Not finished episodes. Consider increasing warmup time'
    if batch_idxs is None:
      batch_idxs = np.random.permutation(len(self.steps))
    experiences = []
    while len(experiences) < batch_size:
      for idx in batch_idxs:
        experiences.append(self.steps[idx])
        if len(experiences) == batch_size:
          break
    assert len(experiences) == batch_size
    return experiences

  def append(self, observation, action, reward, terminal, training=True, pred_action=None):
    step = EpisodeStep(observation, action, reward, terminal, pred_action)
    self.current_episode.append(step, training)
    if terminal:
      self.steps.extend(self.current_episode.steps)
      self.current_episode = Episode(self.gamma)
