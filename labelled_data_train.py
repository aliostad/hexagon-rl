import numpy as np
from dqn_centaur_ai_gym import *
import os
import glob

def one_hot_vector(idx, length):
  a = np.zeros(length)
  a[idx] = 1.
  return a

def build_attack_vector(fileName, discrete=False):
  """

  :type fileName: str
  :return:
  """
  x = np.load(fileName)
  y = np.load(fileName.replace('_STATE_', '_ACTION_'))
  y = np.reshape(y.flatten(), EnvDef.SPATIAL_OUTPUT)
  idx = np.argmax(y.flat)
  if not discrete:
    return np.reshape(x, EnvDef.SPATIAL_INPUT + (1,)), one_hot_vector(idx, EnvDef.SPATIAL_OUTPUT[0])
  _y = idx % EnvDef.MAX_GRID_LENGTH
  _x = idx / EnvDef.MAX_GRID_LENGTH
  return np.reshape(x, EnvDef.SPATIAL_INPUT + (1,)), \
         [one_hot_vector(_x, EnvDef.MAX_GRID_LENGTH),
          one_hot_vector(_y, EnvDef.MAX_GRID_LENGTH)]



def build_decision_vector(fileName):
  """

  :type fileName: str
  :return:
  """
  x = np.load(fileName)
  tail, name = os.path.split(fileName)
  fileNameNoExt = name.replace('.npy', '')
  index = 1 if bool(fileNameNoExt.split('_')[-1]) else 0
  y = np.zeros(EnvDef.DECISION_ACTION_SPACE)
  y[index] = 1
  return x, y


if __name__ == '__main__':

  dec_model = DecisionModel()
  attack_model = AttackModel()
  pattern = None
  vectoriser = None
  model = None

  if len(sys.argv) < 3:
    print('Usage: python labelled_data_train.py <DECISION|ATTACK> <folder>')
    exit(0)
  elif sys.argv[1] == 'DECISION':
    pattern = '/BOOST_VECTOR_*.npy'
    model = dec_model
    vectoriser = build_decision_vector
  elif sys.argv[1] == 'ATTACK':
    pattern = '/ATTACK_STATE_*.npy'
    model = attack_model
    vectoriser = build_attack_vector

  folder = sys.argv[2]

  X = []
  Y = []
  Y_x = []
  Y_y = []
  i = 0
  for f in glob.glob(folder + pattern):
    i += 1
    if i % 100 == 0:
      print(i)
    x, y = vectoriser(f)
    X.append(x)
    Y.append(y)
    if len(y) == 2:
      Y_x.append(y[0])
      Y_y.append(y[1])

  X = np.array(X)
  Y = np.array(Y)
  Y_x = np.array(Y_x)
  Y_y = np.array(Y_y)
  if len(Y_x) > 0:
    Y = [Y_x, Y_y]

  model.model.fit(X, Y, batch_size=100, epochs=200, verbose=1)
  model.model.save(model.modelName)
