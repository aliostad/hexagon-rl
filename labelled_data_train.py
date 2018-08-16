import numpy as np
from super_centaur_ai_gym import *
import os
import glob


def build_attack_vector(fileName):
  """

  :type fileName: str
  :return:
  """
  x = np.load(fileName)
  tail, name = os.path.split(fileName)
  fileNameNoExt = name.replace('.npy', '')
  splits = fileNameNoExt.split('_')
  yx = int(splits[-2])
  yy = int(splits[-1])
  y = np.zeros(EnvDef.MAX_GRID_LENGTH)
  y[yx + EnvDef.MAX_GRID_LENGTH/2] = 1
  return x, y


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
    pattern = '/ATTACK_VECTOR_*.npy'
    model = attack_model
    vectoriser = build_attack_vector

  folder = sys.argv[2]

  X = []
  Y = []
  i = 0
  for f in glob.glob(folder + pattern):
    i += 1
    if i % 100 == 0:
      print(i)
    x, y = vectoriser(f)
    X.append(np.array([x]))  # wrapping in another array is due to the way keras-rl ...
    # uses the model expecting another dimension
    Y.append(y)

  X = np.array(X)
  Y = np.array(Y)
  if sys.argv[1] == 'ATTACK':
    X = X.reshape(X.shape[0], EnvDef.SPATIAL_INPUT[0], EnvDef.SPATIAL_INPUT[1], EnvDef.SPATIAL_INPUT[2])

  model.model.fit(X, Y , batch_size=100, epochs=200, verbose=1)
  model.model.save(model.modelName)
