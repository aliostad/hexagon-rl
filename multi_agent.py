from rl.core import Agent, Env, Processor

class MultiEnv(Env):

  def __init__(self, innerEnvs):
    """

    :type innerEnv: dict of Env
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

  def step(self, action):
    return {name: self.inner_envs[name].step(action[name]) for name in self.inner_envs}

  def seed(self, seed=None):
    self._seed = seed
    return [self._seed]


class MultiAgent(Agent):

  @property
  def layers(self):
    pass

  def backward(self, rewards, terminal):
    return {name: self.inner_agents[name].backward(rewards[name]) for name in rewards}

  def compile(self, optimizer, metrics=[]):
    """
    DONT USE. USE DIRECTLY ON YOUR AGENT
    :param optimizer:
    :param metrics:
    :return:
    """
    raise NotImplementedError()

  def load_weights(self, filepath):
    """
    DONT USE. USE DIRECTLY ON YOUR AGENT
    :param filepath:
    :return:
    """
    raise NotImplementedError()

  def save_weights(self, filepath, overwrite=False):
    """
    DONT USE. USE DIRECTLY ON YOUR AGENT
    :param filepath:
    :param overwrite:
    :return:
    """
    raise NotImplementedError()

  def forward(self, observations):
    return {name: self.inner_agents[name].forward(observations[name]) for name in observations}

  def __init__(self, innerAgents):
    """

    :type innerAgents: dict of Agent
    """
    super(MultiAgent, self).__init__()
    self.inner_agents = innerAgents

class MultiProcessor(Processor):

  def __init__(self, innerProcessors):
    """

    :type innerProcessors: dict of Processor
    """
    self.inner_processors = innerProcessors

  def process_observation(self, observations):
    return {name: self.inner_processors[name].process_observation(observations[name]) for name in observations}

  def process_action(self, actions):
    return {name: self.inner_processors[name].process_action(actions[name]) for name in actions}


