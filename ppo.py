from rl.core import *
from episodic_memory import *
from keras import backend as K

class PPOAgent(Agent):

  def __init__(self, nb_actions, actor, critic, memory, observation_shape,
               gamma=.99, batch_size=32, nb_steps_warmup=100,
               train_interval=1, memory_interval=1, custom_model_objects={}, target_model_update=.001,
               masker=None, training_epochs=10, **kwargs):
    super(PPOAgent, self).__init__(**kwargs)

    # Parameters.
    self.nb_actions = nb_actions
    self.nb_steps_warmup = nb_steps_warmup
    self.gamma = gamma
    self.target_model_update = target_model_update
    self.batch_size = batch_size
    self.train_interval = train_interval
    self.memory_interval = memory_interval
    self.custom_model_objects = custom_model_objects
    self.observation_shape = observation_shape
    self.training_epochs = training_epochs

    # Related objects.
    self.actor = actor
    self.critic = critic
    self.memory = memory
    self.dummy_action = np.zeros((1, self.nb_actions))
    self.dummy_value = np.zeros((1, 1))
    self.masker = masker

    # State.
    self.compiled = True
    self.reset_states()
    self.last_masked_raw_action = None
    self.last_raw_action = None
    self.last_observation = None

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

    raw_action = self.actor.predict([ state.reshape((1,) + state.shape),
                            self.dummy_value, self.dummy_value, self.dummy_action])[0]
    masked_raw_action = raw_action if self.masker is None else self.masker.mask(raw_action)
    return masked_raw_action, raw_action

  def forward(self, observation):
    self.last_observation = observation
    masked_raw_action, raw_action = self.select_action(observation)
    self.last_raw_action = raw_action
    self.last_masked_raw_action = masked_raw_action
    return masked_raw_action

  def backward(self, reward, terminal):
    self.memory.append(self.last_observation,
                       self.last_masked_raw_action, reward, terminal,
                       training=self.training,
                       pred_action=self.last_raw_action)

    if self.training and self.step > self.nb_steps_warmup:
      self._run_training()
    return []

  def _run_training(self):
    experiences = self.memory.sample(self.batch_size)
    observations, actions, pred_actions, rewards = ([], [], [], [])
    for idx, e in enumerate(experiences):
      observations.append(e.observation)
      actions.append(e.action)
      pred_actions.append(e.pred_action)
      rewards.append(e.discounted_reward)
    observations = np.array(observations)
    actions = np.array(actions)
    pred_actions = np.array(pred_actions)
    rewards = np.array(rewards)
    old_prediction = pred_actions
    pred_values = self.critic.predict(observations)
    for e in range(self.training_epochs):
      self.actor.train_on_batch([observations, rewards, pred_values, pred_actions], [actions])
    for e in range(self.training_epochs):
      self.critic.train_on_batch([observations], [rewards])

  @staticmethod
  def proximal_policy_optimization_loss(actual_value, predicted_value, old_prediction):
    advantage = actual_value - predicted_value

    def loss(y_true, y_pred):
      prob = K.sum(y_true * y_pred)
      old_prob = K.sum(y_true * old_prediction)
      r = prob / (old_prob + 1e-10)

      return -K.log(prob + 1e-10) * K.mean(
        K.minimum(r * advantage, K.clip(r, min_value=0.8, max_value=1.2) * advantage))

    return loss