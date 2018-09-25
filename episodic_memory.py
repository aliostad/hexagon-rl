from rl.memory import Memory
from collections import deque
import numpy as np
import math

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
    n = len(self.steps)

    if step.terminal:
      self.is_closed = True
      n = len(self.steps)
      for i, s in enumerate(self.steps):
        for j in range(i+1, n):
          s.discounted_reward += self.steps[i].reward * (self.gamma ** j)  # I think it must be self.gamma ** (n-j) but other impl had just j
      return None



class EpisodicMemory(Memory):
  def __init__(self, experience_window_length, reward_decay_gamma=0.99,
               good_episodes_window=100, only_last_episode=False, **kwargs):
    super(EpisodicMemory, self).__init__(window_length=experience_window_length, **kwargs)
    self.current_episode = Episode(reward_decay_gamma)
    self.gamma = reward_decay_gamma
    self.steps = deque(maxlen=experience_window_length)
    self.good_episodes = deque(maxlen=good_episodes_window)
    self.only_last_episode = only_last_episode
    self.last_episode = None

  def manage_finished_episode(self, episode):
    """

    :type episode: Episode
    :return:
    """
    if len(self.good_episodes) > 0:
      min_reward = np.min(map(lambda e: e.reward, self.good_episodes))
      if episode.reward > min_reward:
        self.good_episodes.append(episode)
    else:
      self.good_episodes.append(episode)

  def sample(self, batch_size, batch_idxs=None):
    """

    :param batch_size: int, size of the batch
    :param batch_idxs: array, indices of !!episodes!! to choose from
    :return:
    """
    assert len(self.steps) > 0, 'Not finished episodes. Consider increasing warmup time'
    if self.only_last_episode:
      assert self.last_episode is not None, 'Last episode is None. Make sure at least one episode finished.'
      return self.last_episode.steps

    if batch_idxs is None:
      batch_idxs = np.random.permutation(len(self.steps))
    experiences = []
    if len(self.good_episodes) > 0:
      n_good_ones = batch_size / 2
      e = np.random.choice(self.good_episodes)
      e = Episode()
      if len(e.steps) < n_good_ones:
        for s in e.steps:
          experiences.append(s)
      else:
        e_idx = np.random.permutation(n_good_ones)
        for ix in e_idx:
          experiences.append(e.steps[ix])

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
      self.last_episode = self.current_episode
      self.current_episode = Episode(self.gamma)
