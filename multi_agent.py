from copy import deepcopy
from rl.core import Agent, Env, Processor, History
import numpy as np
from rl.callbacks import (
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer
)

class MultiEnv(Env):

  def __init__(self, innerEnvs):
    """

    :type innerEnv: dict of str: Env
    """
    #super(MultiEnv, self).__init()
    self.inner_envs = innerEnvs

  def render(self, mode='human', close=False):
    """
    DONT USE. USE DIRECTLY ON YOUR AGENT
    :param mode:
    :param close:
    :return:
    """
    raise NotImplementedError()

  def close(self):
    for name in self.inner_envs:
      self.inner_envs[name].close()

  def configure(self, *args, **kwargs):
    """
    DONT USE. USE DIRECTLY ON YOUR ENV
    :param args:
    :param kwargs:
    :return:
    """
    raise NotImplementedError()

  def reset(self):
    return {name: self.inner_envs[name].reset() for name in self.inner_envs}

  def step(self, actions):
    rewards = {}
    obs = {}
    isDone = False
    for name in self.inner_envs:
      observation, r, done, info = self.inner_envs[name].step(actions[name])
      rewards[name] = r
      obs[name] = observation
      isDone = done or isDone
    return obs, rewards, isDone, {}

  def seed(self, seed=None):
    self._seed = seed
    return [self._seed]


class MultiAgent(Agent):

  @property
  def layers(self):
    pass

  def backward(self, rewards, terminal):
    return {name: self.inner_agents[name].backward(rewards[name], terminal) for name in rewards}

  def compile(self, optimizer, metrics=[]):
    """
    DONT USE. USE DIRECTLY ON YOUR AGENT
    :param optimizer:
    :param metrics:
    :return:
    """
    raise NotImplementedError()

  def load_weights(self, filepaths):
    """

    :type filepaths: dict of str: str
    :return:
    """
    for name in filepaths:
      self.inner_agents[name].load_weights(filepaths[name])

  def save_weights(self, filepaths, overwrite=False):
    """

    :type filepath: dict of str: str
    :param overwrite:
    :return:
    """
    for name in filepaths:
      self.inner_agents[name].save_weights(filepaths[name])

  def forward(self, observations):
    return {name: self.inner_agents[name].forward(observations[name]) for name in observations}

  def __init__(self, innerAgents, processor):
    """

    :type innerAgents: dict of str: Agent
    """
    super(MultiAgent, self).__init__(processor)
    self.inner_agents = innerAgents
    self.compiled = True

  def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
          visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
          nb_max_episode_steps=None):
      """Trains the agent on the given environment.

      # Arguments
          env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
          nb_steps (integer): Number of training steps to be performed.
          action_repetition (integer): Number of times the agent repeats the same action without
              observing the environment again. Setting this to a value > 1 can be useful
              if a single action only has a very small effect on the environment.
          callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
              List of callbacks to apply during training. See [callbacks](/callbacks) for details.
          verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
          visualize (boolean): If `True`, the environment is visualized during training. However,
              this is likely going to slow down training significantly and is thus intended to be
              a debugging instrument.
          nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
              of each episode using `start_step_policy`. Notice that this is an upper limit since
              the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
              at the beginning of each episode.
          start_step_policy (`lambda observation: action`): The policy
              to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
          log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
          nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
              automatically resetting the environment. Set to `None` if each episode should run
              (potentially indefinitely) until the environment signals a terminal state.

      # Returns
          A `keras.callbacks.History` instance that recorded the entire training process.
      """
      if not self.compiled:
          raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
      if action_repetition < 1:
          raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

      self.training = True

      callbacks = [] if not callbacks else callbacks[:]

      if verbose == 1:
          callbacks += [TrainIntervalLogger(interval=log_interval)]
      #elif verbose > 1:
          #callbacks += [TrainEpisodeLogger()]
      if visualize:
          callbacks += [Visualizer()]
      history = History()
      callbacks += [history]
      callbacks = CallbackList(callbacks)
      if hasattr(callbacks, 'set_model'):
          callbacks.set_model(self)
      else:
          callbacks._set_model(self)
      callbacks._set_env(env)
      params = {
          'nb_steps': nb_steps,
      }
      if hasattr(callbacks, 'set_params'):
          callbacks.set_params(params)
      else:
          callbacks._set_params(params)
      self._on_train_begin()
      callbacks.on_train_begin()

      episode = np.int16(0)
      self.step = np.int16(0)
      observation = None
      episode_reward = None
      episode_step = None
      did_abort = False
      try:
          while self.step < nb_steps:
              if observation is None:  # start of a new episode
                  callbacks.on_episode_begin(episode)
                  episode_step = np.int16(0)
                  episode_reward = np.float32(0)

                  # Obtain the initial observation by resetting the environment.
                  self.reset_states()
                  observation = deepcopy(env.reset())
                  if self.processor is not None:
                      observation = self.processor.process_observation(observation)
                  assert observation is not None

                  # Perform random starts at beginning of episode and do not record them into the experience.
                  # This slightly changes the start position between games.
                  nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
                  for _ in range(nb_random_start_steps):
                      if start_step_policy is None:
                          action = env.action_space.sample()
                      else:
                          action = start_step_policy(observation)
                      if self.processor is not None:
                          action = self.processor.process_action(action)
                      callbacks.on_action_begin(action)
                      observation, reward, done, info = env.step(action)
                      observation = deepcopy(observation)
                      if self.processor is not None:
                          observation, reward, done, info = self.processor.process_step(observation, reward, done, info)
                      callbacks.on_action_end(action)
                      if done:
                          warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
                          observation = deepcopy(env.reset())
                          if self.processor is not None:
                              observation = self.processor.process_observation(observation)
                          break

              # At this point, we expect to be fully initialized.
              assert episode_reward is not None
              assert episode_step is not None
              assert observation is not None

              # Run a single step.
              callbacks.on_step_begin(episode_step)
              # This is were all of the work happens. We first perceive and compute the action
              # (forward step) and then use the reward to improve (backward step).
              action = self.forward(observation)
              if self.processor is not None:
                  action = self.processor.process_action(action)
              accumulated_info = {}
              done = False
              callbacks.on_action_begin(action)
              observation, r, done, info = env.step(action)
              observation = deepcopy(observation)
              if self.processor is not None:
                  observation, r, done, info = self.processor.process_step(observation, r, done, info)
              for key, value in info.items():
                  if not np.isreal(value):
                      continue
                  if key not in accumulated_info:
                      accumulated_info[key] = np.zeros_like(value)
                  accumulated_info[key] += value
              callbacks.on_action_end(action)
              reward = r
              if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                  # Force a terminal state.
                  done = True
              metrics = self.backward(reward, terminal=done)
              if isinstance(reward, dict):
                episode_reward = sum(reward.values())
              else:
                episode_reward += reward

              step_logs = {
                  'action': action,
                  'observation': observation,
                  'reward': reward,
                  'metrics': metrics,
                  'episode': episode,
                  'info': accumulated_info,
              }
              callbacks.on_step_end(episode_step, step_logs)
              episode_step += 1
              self.step += 1

              if done:
                  # We are in a terminal state but the agent hasn't yet seen it. We therefore
                  # perform one more forward-backward call and simply ignore the action before
                  # resetting the environment. We need to pass in `terminal=False` here since
                  # the *next* state, that is the state of the newly reset environment, is
                  # always non-terminal by convention.
                  self.forward(observation)
                  self.backward({}, terminal=False)

                  # This episode is finished, report and reset.
                  episode_logs = {
                      'episode_reward': episode_reward,
                      'nb_episode_steps': episode_step,
                      'nb_steps': self.step,
                  }
                  callbacks.on_episode_end(episode, episode_logs)

                  episode += 1
                  observation = None
                  episode_step = None
                  episode_reward = None
      except KeyboardInterrupt:
          # We catch keyboard interrupts here so that training can be be safely aborted.
          # This is so common that we've built this right into this function, which ensures that
          # the `on_train_end` method is properly called.
          did_abort = True
      callbacks.on_train_end(logs={'did_abort': did_abort})
      self._on_train_end()

      return history


class MultiProcessor(Processor):

  def __init__(self, innerProcessors):
    """

    :type innerProcessors: dict of str: Processor
    """
    self.inner_processors = innerProcessors

  def process_observation(self, observation):
    """
    Turns a single observation into many observations based on what each processor needs
    :param observation:
    :return:
    """
    return {name: self.inner_processors[name].process_observation(observation) for name in self.inner_processors}

  def process_action(self, actions):
     return {name: self.inner_processors[name].process_action(actions[name]) for name in actions}



