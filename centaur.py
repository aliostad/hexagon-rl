from hexagon_agent import *

import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation
from collections import deque
from keras import backend as K

K.clear_session()

import tensorflow as tf
graph = None

class Round:
  def __init__(self, world, inputVector, action):
    self.world = world
    self.inputVector = inputVector
    self.action = action

class Centaur(Aliostad):
  def __init__(self, name, modelName):
    Aliostad.__init__(self, name)
    self.modelName = modelName
    self.memory = deque(maxlen=2000)

    self.gamma = 0.85
    self.epsilon = 1.0
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.learning_rate = 0.005
    self.tau = .125
    self.r = random.Random()

    self.MAX_NODES = 10000
    self.NODE_FEATURE_COUNT = 5
    self.ACTION_SPACE = 2
    self.previous = None
    self.version = 1

    self.model = self.create_model()
    self.model._make_predict_function()
    global graph
    graph = tf.get_default_graph()

  def getRandomAction(self):
    if self.r.uniform(0.0, 0.1) < 0.5:
      return [1, 0]
    else:
      return [0, 1]

  def reset(self):
    self.memory = deque(maxlen=2000)
    self.version += 1
    self.save_model(self.modelName + '_' + str(self.version))

  def create_model(self):
    model = Sequential()
    model.add(Dense(48, input_shape=(self.MAX_NODES * self.NODE_FEATURE_COUNT, ), activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(self.ACTION_SPACE))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy",
                  optimizer='adadelta', metrics=['accuracy'])
    return model

  def act(self, inpt):
    self.epsilon *= self.epsilon_decay
    self.epsilon = max(self.epsilon_min, self.epsilon)
    if np.random.random() < self.epsilon:
      return self.getRandomAction()

    global graph
    with graph.as_default():
      action = self.model.predict(inpt)[0]
    return action

  def remember(self, state, action, reward, new_state):
    self.memory.append([state, action, reward, new_state])

  def replay(self):
    global graph
    batch_size = 32
    if len(self.memory) < batch_size:
      return
    samples = random.sample(self.memory, batch_size)
    for sample in samples:
      state, action, reward, new_state = sample
      with graph.as_default():
        target = self.model.predict(state)
        Q_future = max(self.model.predict(new_state)[0])
        target[0][action] = reward + Q_future * self.gamma
        self.model.fit(state, target, epochs=1, verbose=0)

  def save_model(self, fn):
    self.model.save(fn)


  def calculateScoreForRound(self, cells, round_no):
    cnt = len(cells)
    sumz = sum(map(lambda x: x.resources, cells.values()), 1)
    return cnt + math.log(sumz, 3) - math.log(round_no, 3)

  def buildInput(self, world):
    inpt = [np.array([1, 0, 0, 0, 0]) for i in range(0, self.MAX_NODES)]

    for cell in world.cells.values():
      id = int(abs(hash(cell.id))) % self.MAX_NODES
      inpt[id] = np.array([0, 1, 0, 0, cell.resources])
      for n in cell.neighbours:
        id = int(abs(hash(n.id))) % self.MAX_NODES
        if n.isOwned is None:  # neutral
          inpt[id] = np.array(
            [0, 0, 0, 1, cell.resources])  # resources would be 0 but better just to use resources property
        elif n.isOwned is False:  # enemy
          inpt[id] = np.array([0, 0, 1, 0, cell.resources])
    inpt = np.array(inpt).flatten()
    return np.array([inpt])

  def timeForBoost(self, world):
    '''

    :param world: a world: World
    :return: nothing
    '''
    inpt = self.buildInput(world)
    action = None
    if self.previous is not None:
      previousScore = self.calculateScoreForRound(self.previous.world.cells, self.round_no)
      newScore = self.calculateScoreForRound(world.cells, self.round_no)
      reward = newScore - previousScore - len(self.previous.world.cells)  # because every turn a cell gets 1 resource
      action = np.argmax(self.act(inpt))
      self.remember(self.previous.inputVector, action, reward, inpt)
      self.replay()  # internally iterates default (prediction) model
    else:
      action = self.getRandomAction()
    self.previous = Round(world, inpt, action)
    res = action == 1
    print res
    return res

if __name__ == "__main__":
  c = Centaur('cc', 'siose')
  for i in range(0, 1000):
    t = c.turn([
      {
        'id': '0_0',
        'resourceCount': 100,
        'neighbours': [
          {'id': '50_1', 'resourceCount': 0, 'owned': None},
          {'id': '40_1', 'resourceCount': 10, 'owned': False},
          {'id': '30_1', 'resourceCount': 10, 'owned': False},
          {'id': '10_1', 'resourceCount': 10, 'owned': False},
          {'id': '0_60', 'resourceCount': 100, 'owned': True},
          {'id': '0_11', 'resourceCount': 10, 'owned': False}
        ]
      },
      {
        'id': '0_60',
        'resourceCount': 100,
        'neighbours': [
          {'id': '50_11', 'resourceCount': 10, 'owned': False},
          {'id': '40_11', 'resourceCount': 50, 'owned': False},
          {'id': '30_11', 'resourceCount': 30, 'owned': False},
          {'id': '10_11', 'resourceCount': 20, 'owned': False},
          {'id': '0_121', 'resourceCount': 10, 'owned': False},
          {'id': '0_0', 'resourceCount': 100, 'owned': True}
        ]
      }
    ])
    print(t)
    print i