from super_centaur_ai_gym import *
from hexagon_gaming import *
import hexagon_ui_api
from random import Random
import os

# __________________________________________________________________________________________________________________________
class DummySuperCentaurPlayer(Aliostad):
  def __init__(self, name, folder):
    Aliostad.__init__(self, name)
    self.decision_prc = CentaurDecisionProcessor()
    self.attack_processor = CentaurAttackProcessor()
    self.r = Random()
    self.folder = folder

  def timeForBoost(self, world):
    """

    :type world: World
    :return:
    """
    isTime = Aliostad.timeForBoost(self, world)
    vector = self.decision_prc.process_observation(world)
    fileName = '{}/BOOST_VECTOR_{}_{}.npy'.format(self.folder, r.randint(1, 1000 * 1000 * 1000), isTime)
    np.save(fileName, vector)
    return isTime

  def getAttackFromCellId(self, world):
    """

    :type world: World
    :return:
    """
    fromId = Aliostad.getAttackFromCellId(self, world)
    if fromId is not None:
      vector = self.attack_processor.process_observation(world)
      cells = {str(id): world.uberCells[id] for id in world.uberCells}
      sortedCellNames = sorted(cells)
      index = -1
      for i in range(0, len(sortedCellNames)):
        if sortedCellNames[i] == str(fromId):
          index = i
          break
      if index < 0:
        print('Catastrophe!')
        raise IndexError()

      fileName = '{}/ATTACK_VECTOR_{}_{}.npy'.format(self.folder, r.randint(1, 1000 * 1000 * 1000), index)
      np.save(fileName, vector)
    return fromId

# ____________________________________________________________________________________________________________________________

class DataExtractionGym:
  def __init__(self, folder):
    self.players = [
      Aliostad('random80', 0.8),
      Aliostad('random60', 0.6),
      Aliostad('random50', 0.3),
      Aliostad('Ali-1'),
      DummySuperCentaurPlayer('dumm1', folder),
      DummySuperCentaurPlayer('dumm2', folder)
    ]
    self.game = None

  def start(self, episodes=1000, max_rounds=2000):
    self.game = Game('my-game', self.players, r.choice([7, 8, 9, 10]))
    self.game.start()
    for ep in range(0, episodes):
      print('episode {}'.format(ep))
      for i in range(0, max_rounds):
        stats, finished = self.game.run_sync()
        if finished:
          break


if __name__ == '__main__':
  folderName = 'train_data'
  if not os.path.exists(folderName):
    os.makedirs(folderName)

  gym = DataExtractionGym(folderName)
  gym.start()
