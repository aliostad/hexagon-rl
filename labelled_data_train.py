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
  index = int(fileNameNoExt.split('_')[-1])
  y = np.zeros(EnvDef.ATTACK_ACTION_SPACE)
  y[index] = 1
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
  for f in glob.glob(folder + pattern):
    x, y = vectoriser(f)
    X.append(np.array([x]))  # wrapping in another array is due to the way keras-rl ...
    # uses the model expecting another dimension
    Y.append(y)

  model.model.fit(np.array(X),
                  np.array(Y), batch_size=100, epochs=20, verbose=1)
  model.model.save(model.modelName)
