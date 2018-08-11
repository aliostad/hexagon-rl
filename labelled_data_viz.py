import numpy as np
from super_centaur_ai_gym import *
import os
import glob
import sys

def get_attack_potential(v):
  """
      self.attackPotential = self.resources * \
                           math.sqrt(max(self.resources - safeMin([n.resources for n in self.noneOwns]), 1)) / \
                           math.log(sum([n.resources for n in self.enemies], 1) + 1, 5)
    self.canAttack = any(filter(lambda x: x.resources < self.resources + 2, self.enemies))
    srt = sorted(world.uberCells, key=lambda x:
                 -100 if not world.uberCells[x].canAttack else world.uberCells[x].attackPotential *
                 (r.uniform(1.0, 3.0) if self.random_variation else 1)
                 , reverse=True)


      0 - Resources
      1 - Count of friendly neighbouring cells
      2 - Sum of friendly neighbouring cell resources
      3 - Min of friendly neighbouring cell resources
      4 - Max of friendly neighbouring cell resources
      5 - Count of foe neighbouring cell resources
      6 - Sum of foe neighbouring cell resources
      7 - Min of foe neighbouring cell resources
      8 - Max of foe neighbouring cell resources
      9 - Count of neutral

  return -100 if (v[9]==0 and v[5]==0) else v[0] * \
            math.sqrt(max(v[0] - 0 if v[9] > 0 else v[0] - v[7], 1)) \
                                            / math.log(v[6] + 2, 5)

  :param v:
  :return:
  """
  return -100 if (v[1]==0) else v[0] * \
            math.sqrt(max(v[0] - v[2], 1)) \
                                            / math.log(v[2] + 2, 5)


if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Usage: python labelled_data_viz.py <attack vector file>')
    exit(0)

  fname = sys.argv[1]
  x = np.load(fname)
  tail, name = os.path.split(fname)
  fileNameNoExt = name.replace('.npy', '')
  index = int(fileNameNoExt.split('_')[-1])

  for i in range(0, EnvDef.MAX_CELL_COUNT):
    v = x[i*EnvDef.ATTACK_VECTOR_SIZE:(i+1)*EnvDef.ATTACK_VECTOR_SIZE]
    if sum(v) > 0:
      if i == index:
        print '*',
      for item in v:
        print '\t' + str(item),
      print '\t' + str(get_attack_potential(v))
