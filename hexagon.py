import math
from random import Random

r = Random()


def safeMax(list, default=0):
  return default if len(list) == 0 else max(list)

def safeMin(list, default=0):
  return default if len(list) == 0 else min(list)

class Klass:

  def __init__(self, j):
    self.__dict__ = j

class Transaction:

  def __init__(self, fromCell, toCell, resources):
    self.toCell = toCell
    self.fromCell = fromCell
    self.resources = resources

class NeighbourCell:

  def __init__(self, id, isOwned, resources):
    self.isOwned = isOwned
    self.id = id
    self.resources = resources


class Strategy:
  Non = 0
  Defend = 1
  Attack = 2
  Boost = 3
  Expand = 4


class MyCell:

  def __init__(self, id, neighbours, resources):
    '''

    :param id: the id: String
    :param neighbours: ze ns: List
    :param resources: ze rezorces: Int
    '''
    self.resources = resources
    self.id = id
    self.neighbours = neighbours

class TurnStat:

  def __init__(self, strategy=None, cellCount=0, resources=0, negative=0, resourceLossStreak=0):
    self.resources = resources
    self.cellCount = cellCount
    self.strategy = strategy
    self.negative = negative
    self.resourceLossStreak = resourceLossStreak

class UberCell:

  def __init__(self, cell, world):
    '''
    creates stuff
    :param cell: the cell: MyCell
    :param world: stuff: World
    '''

    self.id = cell.id
    self.world = world
    self.neighbours = cell.neighbours
    self.resources = cell.resources
    self.nones = filter(lambda x: x.isOwned is None, self.neighbours)
    self.enemies = filter(lambda x: x.isOwned == False, self.neighbours)
    self.owns = filter(lambda x: x.isOwned == True, self.neighbours)
    self.noneOwns = filter(lambda x: x.isOwned is None or x.isOwned == False, self.neighbours)
    self.boostFactor = 1.0 * math.log(sum((n.resources for n in self.neighbours), 1), 3) * \
                        sum((n.resources for n in self.enemies), 0) * \
                        safeMax([n.resources for n in self.enemies]) * \
                        math.log(max(sum((n.resources for n in self.enemies), 1), 1), 5) / (self.resources + 1)
    self.powerFactor = self.resources / (safeMin([n.resources for n in self.noneOwns]) + 1)
    self.expansionPotential = len(self.nones) * 100
    self.attackPotential = self.resources * \
                           math.log(max(self.resources - safeMin([n.resources for n in self.noneOwns]), 1), 5) / \
                                  math.log(sum([n.resources for n in self.enemies], 1) + 1, 5)
    self.canAttack = any(self.enemies)
    self.canAcceptTransfer = len(self.owns) > 0
    self.depth = 0

  def calculateDepth2(self):
    self.depth = (len(self.owns) * 100) + sum(len(self.world.uberCells[n.id].owns) for n in self.owns)

  def getGivingBoostSuitability(self):
    return self.depth * math.log(self.resources + 1) * (1.7 if self.resources == 100 else 1)

class World:

  def getOfType(self, type):
    dic = {}
    for id in self.cells:
      for n in self.cells[id].neighbours:
        if n.isOwned == type:
          dic[n.id] = None
    return len(dic)

  def buildNeighbourhood(self):
    '''
    build dictionary of how many non-owns cells around cells that we own
    :return:
    '''
    dic = {}
    for cid in self.cells:
      for n in self.cells[cid].neighbours:
        if n.isOwned is None or n.isOwned == False:
          if n.id in dic:
            dic[n.id] += 1
          else:
            dic[n.id] = 1
    return dic

  def __init__(self, cells):

    ccells = []
    for x in cells:
      nn = []
      for n in x['neighbours']:
        nn.append(NeighbourCell(n['id'], n['owned'], n['resourceCount']))
      c = MyCell(x['id'], nn, x['resourceCount'])
      ccells.append(c)

    self.cells = {x.id: x for x in ccells}
    self.uberCells = {x.id: UberCell(x, self) for x in ccells}
    self.resources = sum(self.cells[x].resources for x in self.cells)
    self.cellCount = len(cells)
    self.enemyCounts = self.getOfType(False)
    self.noneCounts = self.getOfType(None)
    self.neighbourhoodCounts = self.buildNeighbourhood()
    for cid in self.uberCells:
      self.uberCells[cid].calculateDepth2()

class Aliostad:

  def __init__(self, name):
    self.name = name
    self.turnNumber = 0
    self.history = []
    self.f = open(name + ".log", mode='w')

  def getEarlyExpansion(self, world):
    '''

    :param self:
    :param world: the world: World
    :return: a tran: Transaction
    '''

    srt = sorted(world.uberCells, key=lambda x: world.uberCells[x].expansionPotential, reverse=True)
    fromCell = srt[0]
    srt2 = sorted(world.uberCells[fromCell].nones, key=lambda x: world.neighbourhoodCounts[x.id], reverse=True)
    toCell = srt2[0]
    return Transaction(fromCell, toCell.id, int(world.uberCells[fromCell].resources * 51 / 100))

  def getBoost(self, world):
    '''

    :param self:
    :param world: a wo: World
    :return: tran: Transaction
    '''
    srt = sorted(world.uberCells, key=lambda x: world.uberCells[x].getGivingBoostSuitability(), reverse=True)
    cellFromId = srt[0]
    cellFrom = world.uberCells[cellFromId]
    srt2= sorted(world.uberCells, key=lambda x:
                 -1000 if not world.uberCells[x].canAcceptTransfer else world.uberCells[x].boostFactor, reverse=True)
    cellToId = srt2[0]
    amount = int(cellFrom.resources * 70 / 100)
    #print "{}: Boost from {} to {}".format(self.name, cellFrom.id, cellToId)
    return Transaction(cellFrom.id, cellToId, amount)

  def getAttack(self, world):
    '''

    :param self:
    :param world: the world: World
    :return: tran: Transaction
    '''


    srt = sorted(world.uberCells, key=lambda x:
    -100 if not world.uberCells[x].canAttack else world.uberCells[x].attackPotential * r.uniform(1.0, 5.0)
                 , reverse=True)
    cellFromId = srt[0]
    cellFrom = world.uberCells[cellFromId]
    srt2 = sorted(cellFrom.enemies, key=lambda x: x.resources * r.uniform(0.1, 05))
    cellTo = srt2[0]
    amount = cellTo.resources + ((cellFrom.resources - cellTo.resources) * 70 / 100)
    #print "{}: Attack from {} to {}".format(self.name, cellFrom.id, cellTo.id)
    return Transaction(cellFrom.id, cellTo.id, amount)

  def timeForBoost(self, world):
    '''
_history.Take(Convert.ToInt32(Math.Log(TurnNumber * 10, 5.8)))
                    .Count(x => x.Strategy == Strateg.Attack) > 1
    :param world: a world: World
    :return: res: Boolean
    '''

    goBack = int(math.sqrt((self.turnNumber+1) * 10))
    count = 0
    for i in range(0, goBack):
      if self.history[-i].strategy == Strategy.Attack:
        count +=1
        if count > 2:
          return True
    return False

  def turnx(self, cells):
    '''

    :param cells: asdas : []
    :return: a tran: Transaction
    '''

    if len(self.history) == 0:
      self.history.append(TurnStat(Strategy.Expand))

    self.turnNumber += 1
    world = World(cells)
    stat = TurnStat(cellCount=world.cellCount, resources=world.resources,
                                 resourceLossStreak=self.history[-1].resourceLossStreak)

    #print "{} ({}) => {}".format(self.name, self.turnNumber, self.history[-1].strategy)

    if self.history[-1].resources > stat.resources:
      stat.resourceLossStreak += 1
    else:
      stat.resourceLossStreak = int(math.sqrt(stat.resourceLossStreak))

    self.history.append(stat)

    if world.noneCounts > 0 and (world.noneCounts * 8 > world.enemyCounts or r.uniform(0, 1) > 0.9):
      stat.strategy = Strategy.Expand
      t = self.getEarlyExpansion(world)
      stat.strategy = Strategy.Expand
      return t

    islands = filter(lambda x: world.neighbourhoodCounts[x] == 6, world.neighbourhoodCounts)
    if len(islands) > 0:
      # build a bridge
      islandId = islands[0]
      candidateFromCells = []
      for c in world.uberCells.values():
        if c.id != islandId:
          for n in c.neighbours:
            if any(filter(lambda x: x.id == n.id, c.neighbours)):
              diff = c.resources - n.resources
              candid = (c.id, n.id, diff, n.resources + (diff * 61 /100))
              candidateFromCells.append(candid)

      srt = sorted(candidateFromCells, key=lambda x: x[2], reverse=True)

      if any(srt):
        cand = srt[0]
        stat.strategy = Strategy.Attack
        return Transaction(cand[0], cand[1], cand[3])

    canAcceptCount = len(filter(lambda x: world.uberCells[x].canAcceptTransfer, world.uberCells))
    if canAcceptCount == 0:
      stat.strategy = Strategy.Attack
      return self.getAttack(world)
    elif stat.resourceLossStreak > 3 or len(filter(lambda x: world.uberCells[x].canAttack,
                          world.uberCells)) == 0 or self.timeForBoost(world):
      stat.strategy = Strategy.Boost
      return self.getBoost(world)
    else:
      stat.strategy = Strategy.Attack
      return self.getAttack(world)

  def turn(self, cells):
    move = self.turnx(cells)
    h = self.history[0]
    self.f.write("{} - {}: From {} to {} with {} \n".format(self.turnNumber,
              h.strategy, move.fromCell, move.toCell, move.resources))
    return move
