from dqn_centaur_ai_gym import *
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
    in_vector = self.attack_processor.buildInput(world)
    out_vector = self.attack_processor.buildOutput(world)
    rid = r.randint(1, 1000 * 1000 * 1000)
    stateFileName = '{}/ATTACK_STATE_{}.npy'.format(self.folder, rid)
    actionFileName = '{}/ATTACK_ACTION_{}.npy'.format(self.folder, rid)
    np.save(actionFileName, out_vector)
    np.save(stateFileName, in_vector)
    return Aliostad.getAttackFromCellId(self, world)

# ____________________________________________________________________________________________________________________________

class DataExtractionGym:
  def __init__(self, folder):
    self.game = None
    self.folder = folder

  def start(self, episodes=1000, max_rounds=2000):
    for ep in range(0, episodes):
      players = [
        Aliostad('random80', 0.8),
        Aliostad('random60', 0.6),
        Aliostad('random50', 0.3),
        Aliostad('Ali-1'),
        DummySuperCentaurPlayer('dumm1', self.folder),
        DummySuperCentaurPlayer('dumm2', self.folder)]
      self.game = Game('my-game', players, r.choice([7, 8, 9, 10]))
      self.game.start()
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
