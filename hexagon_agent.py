from random import Random
from hexagon_gaming import *
from hexagon import *

r = Random()


def safeMax(list, default=0):
  return default if len(list) == 0 else max(list)


def safeMin(list, default=0):
  return default if len(list) == 0 else min(list)


class Klass:

  def __init__(self, j):
    self.__dict__ = j


def translateOwnership(owned):
  if owned is None:
    return 'N'
  if owned:
    return 'O'
  else:
    return 'E'


class Strategy:
  Non = "None"
  Defend = "Defend"
  Attack = "Attack"
  Boost = "Boost"
  Expand = "Expand"
  Island = "Island"


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

    # how suitable is a cell for receiving boost
    #
    self.boostFactor = math.sqrt(sum((n.resources for n in self.enemies), 1)) * \
                       safeMax([n.resources for n in self.enemies], 1) / (self.resources + 1)
    self.powerFactor = self.resources / (safeMin([n.resources for n in self.noneOwns]) + 1)
    self.expansionPotential = len(self.nones) * 100
    self.attackPotential = self.resources * \
                           math.sqrt(max(self.resources - safeMin([n.resources for n in self.noneOwns]), 1)) / \
                           math.log(sum([n.resources for n in self.enemies], 1) + 1, 5)
    self.hasEnemyNeighbours = any(self.enemies)
    self.canAttack = any(filter(lambda x: x.resources < self.resources + 2, self.enemies))
    self.canAttackOrExpand = any(filter(lambda x: x.resources < self.resources + 2, self.noneOwns))
    self.canAcceptTransfer = len(self.owns) > 0  # don't send to islands, no point (could be?)
    self.depth = 0

  def calculateDepth2(self):
    self.depth = (len(self.owns) * 100) + sum(len(self.world.uberCells[n.id].owns) for n in self.owns)

  def getGivingBoostSuitability(self):
    return (self.depth + 1) * math.sqrt(self.resources + 1) * (1.7 if self.resources == 100 else 1)


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
        if n.isOwned is None or n.isOwned is False:
          if n.id in dic:
            dic[cid] += 1
          else:
            dic[cid] = 1
    return dic

  @staticmethod
  def buildWorldmap(cells):
    """

    :type cells: list of CellInfo
    :return:
    """
    worldmap = {}
    for c in cells:
      if c.id not in worldmap:
        worldmap[c.id] = c.resources
      for n in c.neighbours:
        if n.id not in worldmap:
          worldmap[n.id] = (1 if n.isOwned else -1) * n.resources
    return worldmap

  def buildNonOwnsNeighbourhood(self):
    '''
    build dictionary of how many owned cells around non-owned cells so can be attacked easier
    :return:
    '''
    dic = {}
    for cid in self.cells:
      for n in self.cells[cid].neighbours:
        if n.isOwned is None or n.isOwned is False:
          if n.id in dic:
            dic[n.id] += 1
          else:
            dic[n.id] = 1
    return dic

  def __init__(self, cells):
    """

    :type cells: list of CellInfo
    """
    self.worldmap = World.buildWorldmap(cells)
    self.cells = {x.id: x for x in cells}
    self.uberCells = {x.id: UberCell(x, self) for x in cells}
    self.resources = sum(self.cells[x].resources for x in self.cells)
    self.cellCount = len(cells)
    self.enemyCounts = self.getOfType(False)
    self.noneCounts = self.getOfType(None)
    self.neighbourhoodCounts = self.buildNeighbourhood()
    self.neighbourhoodNonOwnCounts = self.buildNonOwnsNeighbourhood()

    for cid in self.uberCells:
      self.uberCells[cid].calculateDepth2()


class Aliostad(Player):

  def __init__(self, name, randomBoostFactor=None, randomVariation=False):
    Player.__init__(self, name)
    self.round_no = 0
    self.history = []
    self.random_boost = randomBoostFactor
    self.f = open(name + ".log", mode='w')
    self.random_variation = randomVariation

  @staticmethod
  def transform_jsoncells_to_infos(cells):
    ccells = []
    for x in cells:
      nn = []
      for n in x['neighbours']:
        nn.append(NeighbourInfo(n['id'], n['resourceCount'], n['owned']))
      c = CellInfo(x['id'], x['resourceCount'], nn)
      ccells.append(c)
    return ccells

  def getEarlyExpansion(self, world):
    '''

    :param self:
    :param world: the world: World
    :return: a tran: Transaction
    '''

    srt = sorted(world.uberCells, key=lambda x: world.uberCells[x].expansionPotential, reverse=True)
    fromCell = srt[0]
    srt2 = sorted(world.uberCells[fromCell].nones, key=lambda x: world.neighbourhoodNonOwnCounts[x.id], reverse=True)
    toCell = srt2[0]
    return Move(fromCell, toCell.id, int(world.uberCells[fromCell].resources * 51 / 100))

  def getBoost(self, world):
    '''

    :param self:
    :param world: a wo: World
    :return: tran: Transaction
    '''
    srt = sorted(world.uberCells, key=lambda x: world.uberCells[x].getGivingBoostSuitability(), reverse=True)
    cellFromId = srt[0]
    cellFrom = world.uberCells[cellFromId]
    srt2 = sorted(world.uberCells, key=lambda x:
    -1000 if not world.uberCells[x].canAcceptTransfer or
             x == cellFromId else world.uberCells[x].boostFactor, reverse=True)

    cellToId = srt2[0]
    amount = int(cellFrom.resources * 70 / 100)
    return Move(cellFrom.id, cellToId, amount)

  def getAttackFromCellId(self, world):
    """

    :type world: World
    :return:
    """
    srt = sorted(world.uberCells, key=lambda x:
                 -100 if not world.uberCells[x].canAttack else world.uberCells[x].attackPotential *
                 (r.uniform(1.0, 3.0) if self.random_variation else 1)
                 , reverse=True)

    return None if len(srt) == 0 else srt[0]

  def getAttack(self, world):
    '''

    :param self:
    :param world: the world: World
    :return: tran: Transaction
    '''

    cellFromId = self.getAttackFromCellId(world)

    if cellFromId is None:
      return Move(CellId(0, 0), CellId(0, 0), 1000)  # invalid move, nothing better to do

    cellFrom = world.uberCells[cellFromId]
    srt2 = sorted(cellFrom.enemies, key=lambda x: x.resources *
                                                  (r.uniform(0.1, 05) if self.random_variation else 1))
    if len(srt2) == 0:
      return Move(CellId(0, 0), CellId(0, 0), 1000)  # invalid move, nothing better to do
    cellTo = srt2[0]
    amount = cellTo.resources + ((cellFrom.resources - cellTo.resources) * 70 / 100)
    # print "{}: Attack from {} to {}".format(self.name, cellFrom.id, cellTo.id)
    return Move(cellFrom.id, cellTo.id, amount)

  def timeForBoost(self, world):
    '''

    :param world: a world: World
    :return: res: Boolean
    '''
    if self.random_boost is not None:
      return r.uniform(0, 1) < self.random_boost

    goBack = int(math.sqrt(len(world.cells)))
    count = 0
    for i in range(0, goBack):
      if self.history[-i].strategy == Strategy.Attack:
        count += 1
        if count > 4:
          return True
    return False

  def build_world(self, cells):
    """

    :type cells: list of CellInfo
    :return:
    """
    if len(self.history) == 0:
      self.history.append(TurnStat(Strategy.Expand))

    self.round_no += 1
    world = World(cells)
    return world

  def turnx(self, world):
    '''

    :type world: World
    :return: Move
    '''
    stat = TurnStat(cellCount=world.cellCount, resources=world.resources,
                    resourceLossStreak=self.history[-1].resourceLossStreak)

    if self.history[-1].resources > stat.resources:
      stat.resourceLossStreak += 1
    else:
      stat.resourceLossStreak = int(math.sqrt(stat.resourceLossStreak))

    if world.noneCounts > 0 and (world.noneCounts * 8 > world.enemyCounts or
                                 (r.uniform(0, 1) > 0.9 if self.random_variation else False)):
      stat.strategy = Strategy.Expand
      t = self.getEarlyExpansion(world)
      stat.strategy = Strategy.Expand
      return t, stat, world

    islands = filter(lambda x: world.neighbourhoodCounts[x] > 5, world.neighbourhoodCounts)
    if len(islands) > 0:
      # build a bridge
      islandId = islands[0]
      candidateFromCells = []
      for c in world.uberCells.values():
        if c.id != islandId:
          for n in c.neighbours:
            if any(filter(lambda x: x.id == n.id, c.neighbours)):
              diff = c.resources - n.resources
              candid = (c.id, n.id, diff, n.resources + (diff * 61 / 100))
              candidateFromCells.append(candid)

      srt = sorted(candidateFromCells, key=lambda x: x[2], reverse=True)

      if any(srt):
        cand = srt[0]
        stat.strategy = Strategy.Island
        return Move(cand[0], cand[1], cand[3]), stat, world

    canAcceptCount = len(filter(lambda x: world.uberCells[x].canAcceptTransfer, world.uberCells))
    if canAcceptCount == 0:
      stat.strategy = Strategy.Attack
      return self.getAttack(world), stat, world
    elif not any(filter(lambda x: x.canAttack, world.uberCells.values())):
      stat.strategy = Strategy.Boost
      return self.getBoost(world), stat, world
    elif stat.resourceLossStreak > 3 or len(filter(lambda x: world.uberCells[x].canAttack,
                                                   world.uberCells)) == 0 or self.timeForBoost(world):
      stat.strategy = Strategy.Boost
      return self.getBoost(world), stat, world
    else:
      stat.strategy = Strategy.Attack
      return self.getAttack(world), stat, world

  def move(self, playerView):
    """

    :type playerView: PlayerView
    :return: Move
    """
    move, h, world = self.turnx(self.build_world(playerView.ownedCells))
    self.history.append(h)
    self.f.write("{} - {}: From {} to {} with {} - [{}] \n".format(self.round_no,
                                                                   h.strategy,
                                                                   move.fromCell,
                                                                   move.toCell,
                                                                   move.resources,
                                                                   world.cells[move.fromCell]))
    return move

  def movex(self, world):
    """

    :type world: World
    :return: Move
    """
    move, h, worldx = self.turnx(world)
    self.history.append(h)
    self.f.write("{} - {}: From {} to {} with {} \n".format(self.round_no,
                                                            h.strategy,
                                                            move.fromCell,
                                                            move.toCell,
                                                            move.resources))
    return move

  def turn(self, cells):
    seashells = Aliostad.transform_jsoncells_to_infos(cells)
    move, h, world = self.turnx(self.build_world(seashells))
    self.history.append(h)
    self.f.write("{} - {}: From {} to {} with {} - [{}] \n".format(self.round_no,
                                                                   h.strategy,
                                                                   move.fromCell,
                                                                   move.toCell,
                                                                   move.resources,
                                                                   world.cells[move.fromCell]))
    return move
