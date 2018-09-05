from keras.models import Sequential, Model
from keras.layers import Conv2D, Lambda, Input, Flatten
import numpy as np
import math
import sys
from keras.models import load_model
import os
import keras.backend as K

GRID_LENGTH = 9
GRID_SIZE = (GRID_LENGTH, GRID_LENGTH)
RIM_SIZE = 2
SAMPLE_SIZE = 10 * 1000
EPOCHS = 100

def in_between(value, tuple):
  return tuple[0] <= value <= tuple[1]

def build_random_grid():
  treshkhold = (RIM_SIZE, GRID_LENGTH - RIM_SIZE -1)
  return np.array([
    np.array([np.random.uniform(-100, 200) if in_between(i, treshkhold) and in_between(j, treshkhold) else 0 for i in range(0, GRID_LENGTH)])
    for j in range(0, GRID_LENGTH)])

def score_neighbourhood(nn):
  """

  :type nn: ndarray
  :return:
  """
  flat = nn.flatten()
  enemies = filter(lambda x: x < 0, flat)
  if nn[1][1] <= 0:
    return 0
  if min(flat) >= 0:  # no enemy
    return 0
  if -max(enemies) > nn[1][1]+2:  # can't attack
    return min(enemies)
  return nn[1][1] * (nn[1][1] + max(enemies)) / math.log(-sum(enemies) + 1, 5)

def score_neighbourhood_xx(nn):
  """

  :type nn: ndarray
  :return:
  """
  flat = nn.flatten()
  enemies = filter(lambda x: x < 0, flat)
  if nn[1][1] <= 0:
    return 0
  if min(flat) >= 0:  # no enemy
    return 0
  if -max(enemies) > nn[1][1]+2:  # can't attack
    return min(enemies)
  return nn[1][1] * (nn[1][1] + max(enemies)) / (-sum(enemies) + 1)


def score_neighbourhood_x(nn):
  """

  :type nn: ndarray
  :return:
  """
  flat = nn.flatten()
  enemies = filter(lambda x: x < 0, flat)
  if nn[1][1] <= 0:
    return nn[1][1]
  if min(flat) >= 0:  # no enemy
    return 0
  if -max(enemies) > nn[1][1]+2:  # can't attack
    return min(enemies)
  return nn[1][1]


def score_neighbourhood_simple(nn):
  """

  :type nn: ndarray
  :return:
  """
  flat = nn.flatten()
  enemies = filter(lambda x: x < 0, flat)
  maxEnemies = max(enemies) if any(enemies) else 0
  return nn[1][1] + maxEnemies

def score_neighbourhood_simplest(nn):
  """

  :type nn: ndarray
  :return:
  """
  return nn[1][1]

def score_victor(victor):
  res = np.zeros(GRID_SIZE)

  for j in range(RIM_SIZE, GRID_LENGTH-RIM_SIZE):
    for i in range(RIM_SIZE, GRID_LENGTH-RIM_SIZE):
      res[i][j] = score_neighbourhood_xx(victor[i-1:i+2, j-1:j+2])
  return res

class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __str__(self):
    return '{},{}'.format(self.x, self.y)

  def __repr__(self):
    return self.__str__()

def build_model():
  model = Sequential()
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=(GRID_SIZE[0], GRID_SIZE[1], 1)))
  model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(4, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(1, (1, 1), padding='same', activation='relu'))
  return model

def chuz_top(input):
  """

  :type input: ndarray
  :return:
  """
  return K.argmax(input, 1)


if __name__ == '__main__':
  fname = 'scratch_conv.h5f'

  if len(sys.argv) < 2:
    print('use with TRAIN or SCORE')
  elif sys.argv[1] == 'TRAIN':
    model = build_model()
    X = []
    Y = []
    for i in range(0, SAMPLE_SIZE):
      x = build_random_grid()
      y = score_victor(x)
      X.append(x)
      Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(X.shape[0], GRID_SIZE[0], GRID_SIZE[1], 1)
    Y = Y.reshape(X.shape[0], GRID_SIZE[0], GRID_SIZE[1], 1)
    if os.path.exists(fname):
      model.load_weights(fname)
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')
    model.fit(X, Y, batch_size=100, verbose=1, epochs=EPOCHS)
    model.save(fname)
  elif sys.argv[1] == 'SCORE':
    model = load_model(fname)
    xxx = build_random_grid()
    xx = np.array([xxx])
    x = xx.reshape(xx.shape[0], GRID_SIZE[0], GRID_SIZE[1], 1)
    y_hat = model.predict(x)
    #y_hat = y_hat.reshape(GRID_SIZE[0], GRID_SIZE[1])
    print('____________________________')
    y_hat = y_hat.reshape(GRID_SIZE[0], GRID_SIZE[1])
    y_hat_scores = {}
    y = score_victor(xxx)
    y_scores = {}
    for j in range(0, GRID_SIZE[0]):
      for i in range(0, GRID_SIZE[1]):
        y_hat_scores[Point(i, j)] = round(y_hat[i][j])
        y_scores[Point(i, j)] = round(y[i][j])
        print('{} ({}|{})\t'.format(round(xxx[i][j]), round(y[i][j]), round(y_hat[i][j])))

    sorted_y_hat = sorted(y_hat_scores, key=lambda p: y_hat_scores[p], reverse=True)
    sorted_y = sorted(y_scores, key=lambda p: y_scores[p], reverse=True)

    print('TOP 5:')
    for i in range(0, 5):
      print('{} ({}) - {} ({})'.format(sorted_y[i], y_scores[sorted_y[i]], sorted_y_hat[i], y_hat_scores[sorted_y_hat[i]]))

'''
    input_1 = Input(shape=(GRID_SIZE[0], GRID_SIZE[1], 1))
    conv_1 = Conv2D(128, (3, 3), padding='same', activation='relu')
    conv_2 = Conv2D(16, (3, 3), padding='same', activation='relu')
    conv_3 = Conv2D(4, (3, 3), padding='same', activation='relu')
    conv_4 = Conv2D(1, (1, 1), padding='same', activation='relu')
    flat_1 = Flatten()
    lamma = Lambda(chuz_top)
    stack = conv_1(input_1)
    stack = conv_2(stack)
    stack = conv_3(stack)
    stack = conv_4(stack)
    stack = flat_1(stack)
    stack = lamma(stack)
    m2 = Model(inputs=input_1, outputs=stack)
    m2.load_weights(fname)

    print m2.predict(x)
'''