from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Conv1D, Conv2D, MaxPooling1D
from keras.layers.merge import Concatenate
import numpy as np

INPUT_SIZE = 12
SEQUENCE_LENGTH = 120
OUTPUT_SIZE = 2
SAMPLE_SIZE = 100 * 1000

def score_vectore(vector):
  """

  :type vector: list of float
  :return:
  """
  x = 0.
  y = 0.

  for i in range(0, len(vector)):
    if i % 2 == 0:
      if vector[i] > 12:
        x += vector[i]
    else:
      y += vector[i] + min(vector[i-1], vector[i])

  return x, y

def score_victor(vector):
  """

  :type vector: list of list of float
  :return:
  """
  x = np.argmin(vector)
  y = np.argmax(vector)
  return x, y



if __name__ == '__main__':
  model = Sequential()
  #model.add(Conv2D(OUTPUT_SIZE*10, (3, 3), input_shape=(SEQUENCE_LENGTH, INPUT_SIZE, 1)))
  #model.add(Flatten())
  model.add(Dense(OUTPUT_SIZE*8, activation='relu', input_dim=SEQUENCE_LENGTH))
  model.add(Dense(OUTPUT_SIZE*4, activation='relu'))
  model.add(Dense(OUTPUT_SIZE))
  model.compile(loss='mean_squared_error', optimizer='adam')

  X = []
  Y = []
  for i in range(0, SAMPLE_SIZE):
    x =np.array([np.random.uniform(0.3, 68) for j in range(0, SEQUENCE_LENGTH)])
    z, t = score_victor(x)
    y = np.array([z, t])
    X.append(x)
    Y.append(y)

  X = np.array(X)
  Y = np.array(Y)
  #X = X.reshape(X.shape[0], SEQUENCE_LENGTH, INPUT_SIZE, 1)

  model.fit( X, Y, batch_size=100, verbose=1, epochs=100)

