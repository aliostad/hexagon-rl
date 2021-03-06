from rl.core import *
from episodic_memory import *
from keras import backend as K
import os

def exponential_average(old, new, b1):
  return old * b1 + (1-b1) * new

def proximal_policy_optimization_loss_continuous(advantage, old_prediction,
                                                 noise=1., loss_clipping=0.2, epsilon=1e-10):
  def loss(y_true, y_pred):
    var = K.square(noise)
    pi = 3.1415926
    denom = K.sqrt(2 * pi * var)
    prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
    old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

    prob = prob_num/denom
    old_prob = old_prob_num/denom
    r = prob/(old_prob + epsilon)
    return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - loss_clipping,
                                                   max_value=1 + loss_clipping) * advantage))
  return loss

def proximal_policy_optimization_loss(advantage, old_prediction,
                                      entropy_loss=5*1e-3, loss_clipping=0.2, epsilon=1e-10):

  def loss(y_true, y_pred):
    prob = K.sum(y_true * y_pred)
    old_prob = K.sum(y_true * old_prediction)
    r = prob / (old_prob + epsilon)
    entr_loss = (prob * -K.log(K.abs(prob) + epsilon) * entropy_loss)
    clip = K.mean(
      K.minimum(r * advantage, K.clip(r,
                                      min_value=1.-loss_clipping,
                                      max_value=1.+loss_clipping) * advantage))
    rt = entr_loss + clip
    return rt

  return loss

class PPOAgent(Agent):

  default_player = 'default'

  def __init__(self, nb_actions, actor, critic, memories, observation_shape,
               gamma=.99, batch_size=32, nb_steps_warmup=100,
               train_interval=None, memory_interval=1, target_model_update=.001,
               masker=None, training_epochs=10, train_on_last_episode=False,
               noise=1., exploration_ratio=0.99, continuous=False,
               verbose=True, name='default', players=[], **kwargs):
    """

    :param nb_actions:
    :param actor:
    :param critic:
    :param memories:
    :param observation_shape:
    :param gamma:
    :param batch_size:
    :param nb_steps_warmup:
    :param train_interval:
    :param memory_interval:
    :param target_model_update:
    :param masker:
    :param training_epochs:
    :param train_on_last_episode:
    :param noise:
    :param exploration_ratio:
    :param continuous:
    :param verbose:
    :param name:
    :param players:
    :param kwargs:
    """
    super(PPOAgent, self).__init__(**kwargs)
    # Parameters.
    self.nb_actions = nb_actions
    self.nb_steps_warmup = nb_steps_warmup
    self.gamma = gamma
    self.target_model_update = target_model_update
    self.batch_size = batch_size
    self.train_interval = batch_size if train_interval is None else train_interval
    self.memory_interval = memory_interval
    self.observation_shape = observation_shape
    self.training_epochs = training_epochs
    self.train_on_last_episode = train_on_last_episode
    self.verbose = verbose
    self.noise = noise
    self.exploration_ratio = exploration_ratio
    self.continuous = continuous
    self.name = name

    # Related objects.
    self.actor = actor
    self.critic = critic
    if isinstance(memories, dict):
      self.memories = memories
    else:
      self.memories = {PPOAgent.default_player: memories}
    self.dummy_action = np.zeros((1, self.nb_actions))
    self.dummy_value = np.zeros((1, 1))
    self.masker = masker
    self.players = players
    if not any(self.players):
      players.append(PPOAgent.default_player)

    # State.
    self.compiled = True
    self.reset_states()
    self.last_masked_raw_action = None
    self.last_raw_action = None
    self.last_observation = None
    self.last_one_hot_action = None
    self.rewards_over_time = []

  def compile(self, optimizer, metrics=[]):
    raise Exception('Not supporting compilation. Please ensure actor and critic are compiled')

  def load_weights(self, filepath):
    filename, extension = os.path.splitext(filepath)
    actor_filepath = filename + '_actor' + extension
    critic_filepath = filename + '_critic' + extension
    self.actor.load_weights(actor_filepath)
    self.critic.load_weights(critic_filepath)

  def save_weights(self, filepath, overwrite=False):
    filename, extension = os.path.splitext(filepath)
    actor_filepath = filename + '_actor' + extension
    critic_filepath = filename + '_critic' + extension
    self.actor.save_weights(actor_filepath, overwrite=overwrite)
    self.critic.save_weights(critic_filepath, overwrite=overwrite)

  def select_action(self, state):
    """

    :type state: ndarray
    :return:
    """
    raw_action = self.actor.predict([state.reshape((1,) + state.shape),
                            self.dummy_value, self.dummy_action])[0]
    masked_raw_action = raw_action if self.masker is None else self.masker.mask(raw_action)
    if self.training:
      the_choice = np.random.choice(self.nb_actions, p=np.nan_to_num(masked_raw_action))
    else:
      the_choice = np.argmax(masked_raw_action)
    one_hot_action = np.zeros(self.nb_actions)
    one_hot_action[the_choice] = 1.
    return masked_raw_action, raw_action, one_hot_action

  def select_action_continuous(self, state):
    p = self.actor.predict([state.reshape((1,) + state.shape), self.dummy_value, self.dummy_action])
    if self.training and np.random.uniform() < self.exploration_ratio:
      action = action_matrix = p[0] + np.random.normal(loc=0, scale=self.noise, size=p[0].shape)
    else:
      action = action_matrix = p[0]
    return action, action_matrix, p[0]

  def forward(self, observation):
    if self.continuous:
      self.last_observation = observation
      masked_raw_action, raw_action, one_hot_action = self.select_action_continuous(observation)
      self.last_raw_action = raw_action
      self.last_masked_raw_action = masked_raw_action
      self.last_one_hot_action = one_hot_action
      return raw_action
    else:
      self.last_observation = observation
      masked_raw_action, raw_action, one_hot_action = self.select_action(observation)
      self.last_raw_action = raw_action
      self.last_masked_raw_action = masked_raw_action
      self.last_one_hot_action = one_hot_action
      return np.argmax(one_hot_action)

  def backward(self, reward, terminal, player=default_player):
    if player not in self.memories:
      player = PPOAgent.default_player
    self.memories[player].append(self.last_observation,
                       self.last_one_hot_action, reward, terminal,
                       training=self.training,
                       pred_action=self.last_masked_raw_action)

    if self.training and self.train_on_last_episode and terminal:
      self._run_training(player)
    elif self.training and self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
      self._run_training(player)

    if terminal:
      self.rewards_over_time.append(self.memories[player].last_episode.total_reward)  # dodgey. Call beyond interface
      if self.verbose and len(self.rewards_over_time) % 10 == 0:
        print('Average Reward - Last 10:{}\tLast 100:{}\tLast 1000:{}'.format(
          np.average(self.rewards_over_time[-10:]),
          np.average(self.rewards_over_time[-100:]),
          np.average(self.rewards_over_time[-1000:])
        ))
    return []

  def _run_training(self, player):
    if len(self.memories[player].steps) == 0:
      warnings.warn('Memory for {} is empty. cannot train'.format(self.name))
      return # cannot train yet
    else:
      print('OK, here we go again with training {}'.format(player))
    experiences = self.memories[player].sample(self.batch_size)
    observations, actions, pred_actions, rewards = ([], [], [], [])
    for idx, e in enumerate(experiences):
      observations.append(e.observation)
      actions.append(e.action)
      pred_actions.append(e.pred_action)
      rewards.append([e.discounted_reward])
    observations = np.array(observations)
    actions = np.array(actions)
    pred_actions = np.array(pred_actions)
    rewards = np.array(rewards)
    pred_values = self.critic.predict(observations)
    advantages = rewards - pred_values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
    for e in range(self.training_epochs):
      self.actor.train_on_batch([observations, advantages, pred_actions], [actions])
    for e in range(self.training_epochs):
      self.critic.train_on_batch([observations], [rewards])