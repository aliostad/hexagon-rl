from copy import deepcopy
class CellId:
  _poleCache = {}

  def __init__(self, nwes, x):
    '''

    :param nwes: north-west to south-east: int
    :param x: the x: int
    '''
    self.nwes = nwes
    self.x = x
    self.distance_to_centre = max(max(abs(nwes), abs(x)), abs(nwes + x))+1
    self._hash = self.x * 10000 + self.nwes  # this faster
    self._str = '{}_{}'.format(self.nwes, self.x)

  def __str__(self):
    return self._str

  def toJson(self):
    return self.__str__()

  def __hash__(self):
    return self._hash

  def get_opposite(self):
    return CellId(-self.nwes, -self.x)

  def __eq__(self, other):
    if isinstance(other, CellId):
      return other.x == self.x and other.nwes == self.nwes
    return False

  @staticmethod
  def get_poles(radius):
    """

    :type radius: int
    :return: list of CellId
    """
    if radius not in CellId._poleCache:
       CellId._poleCache[radius] = [
          CellId(0, radius - 1),
          CellId(radius - 1, 0),
          CellId(-radius + 1, radius - 1),
          CellId(radius - 1, -radius + 1),
          CellId(-radius + 1, 0),
          CellId(0, -radius + 1)
        ]
    return CellId._poleCache[radius]

  def is_pole(self, radius):
    """

    :type radius: int
    :return: boolean
    """
    return self in CellId.get_poles(radius)


  def get_neighbours(self):
    return [CellId(self.nwes-1, self.x+1),
            CellId(self.nwes+1, self.x-1),
            CellId(self.nwes+1, self.x),
            CellId(self.nwes-1, self.x),
            CellId(self.nwes, self.x+1),
            CellId(self.nwes, self.x-1)]


class CellInfo:
  def __init__(self, id, resources, neighbours):
    """

    :type id: CellId
    :type resources: int
    :type neighbours: list of NeighbourInfo
    """
    self.id = id
    self.resources = resources
    self.neighbours = neighbours


class NeighbourInfo:
  def __init__(self, id, resources, isOwned):
    """

    :type id: CellId
    :type resources:
    :param isOwned: boolean or None
    """
    self.id = id
    self.resources = resources
    self.isOwned = isOwned


class Cell:

  MaximumResource = 100
  NoOwner = ''

  def __init__(self, id, owner, resources, neighbours):
    """
    A view model for sending cell info to players
    :type id: CellId
    :type owner: str
    :type resources: int
    :type neighbours: dict of CellId
    """
    self.id = id
    self.resources = resources
    self.neighbours = neighbours
    self.owner = owner

  def try_transfer_from(self, fromCell, transfer):
    '''

    :type fromCell: Cell
    :type transfer: int
    :return: (success, possible error)
    '''

    if fromCell.id == self.id:
      return False, 'Cannot transfer from {} to itself'.format(fromCell.id)
    if transfer > fromCell.resources:
      return False, 'Cannot transfer {} from {} to {}. It is more than it has ({})'.format(
        transfer, fromCell.id, self.id, fromCell.resources)
    if self.owner != fromCell.owner:
      if self.resources > transfer:
        return False, '{}: Cell {} cannot be captured since it has {} but transfer from {} is only {}'.format(
          fromCell.owner, self.id, self.resources, fromCell.id, transfer)
      elif not self.id in fromCell.neighbours:
        return False, '{}: Cell {} cannot capture {} since they are not neighbours'.format(
          fromCell.owner, fromCell.id, self.id)

    if self.owner == fromCell.owner:
      self.resources += transfer
    else:
      self.resources = transfer - self.resources
    fromCell.resources -= transfer
    self.owner = fromCell.owner

    return True, ''

  def to_neighbour_info(self, owner):
    return NeighbourInfo(self.id, self.resources, None if self.owner == Cell.NoOwner else owner == self.owner)

  def increment_resources(self):
    if self.owner != Cell.NoOwner and self.resources < Cell.MaximumResource:
      self.resources += 1
  def clone(self):
    return Cell(self.id, self.owner, self.resources, deepcopy(self.neighbours))


class BoardSnapshot:
  def __init__(self, cells):
    """

    :type cells: list of Cell
    """
    self.cells = cells


class Board:

  @staticmethod
  def build_ids_for_radius(radius):
    """
    
    :param radius: 
    :return: list of CellId
    """
    r = radius - 1
    l = []
    for row in range(-r, r+1):
      frm = max(-row-r, -r)
      t   = min(r-row, r)
      for nwes in range(frm, t+1):
        l.append(CellId(nwes, row))
    return l

  def __init__(self, radius, cells=None):
    """
    :type radius: int
    :type cells: dict of CellId
    """
    self.radius = radius
    self.diameter = radius*2 - 1
    self.ids = Board.build_ids_for_radius(radius)
    self.cells = {}

    if cells is None:
      for id in self.ids:
        self.cells[id] = Cell(id, Cell.NoOwner, 0, {})

      #add normal neighbours
      for id in self.cells:
        self.cells[id].neighbours = {}.fromkeys(filter(lambda x: x in self.cells, id.get_neighbours()))

      # add edge wrap-around neighbours
      for cid in filter(lambda k: len(self.cells[k].neighbours) < 6, self.cells):
        if cid.is_pole(radius):
          op = cid.get_opposite()
          self.cells[cid].neighbours[op] = None  # add
          for nid in filter(lambda x: x.distance_to_centre == op.distance_to_centre, op.get_neighbours()):
            self.cells[cid].neighbours[nid] = None  # add
        else:
          op = cid.get_opposite()
          self.cells[cid].neighbours[op] = None  # add

      # now sort those having 5. They will end up with 7 neighbours!
      for cid in filter(lambda x: len(self.cells[x].neighbours) < 6, self.cells):
        op = cid.get_opposite()
        for nid in filter(lambda x: x.distance_to_centre == op.distance_to_centre, op.get_neighbours()):
          self.cells[cid].neighbours[nid] = None  # add
    else:
      self.cells = cells

  def get_cell_info(self, id):
    """

    :type id: CellId
    :return: CellInfo
    """
    cell = self.cells[id]
    return CellInfo(cell.id, cell.resources, [self.cells[cid].to_neighbour_info(cell.owner) for cid in cell.neighbours])

  def get_cell_infos_for_player(self, playerName):
    """

    :type playerName: str
    :return: list of CellInfo
    """
    return [self.get_cell_info(cell.id) for cell in
            map(lambda cid: self.cells[cid], filter(lambda cid: self.cells[cid].owner == playerName, self.cells))]

  def get_snapshot(self):
    """

    :return: BoardSnapshot
    """
    return BoardSnapshot(self.cells.values())

  def increment_resources(self):
    for cell in self.cells.values():
      cell.increment_resources()

  def change_ownership(self, id, owner, resources=None):
    """

    :type id: CellId
    :type owner: str
    :type resources: int
    :return:
    """
    self.cells[id].owner = owner
    if resources is not None:
      self.cells[id].resources = resources

  def try_transfer(self, move):
    """

    :type move: Move
    :return: (bool, error)
    """
    if move.fromCell not in self.cells:
      return False, 'Cell does not exist on the board: {}'.format(move.fromCell)
    elif move.toCell not in self.cells:
      return False, 'Cell does not exist on the board: {}'.format(move.toCell)
    return self.cells[move.toCell].try_transfer_from( self.cells[move.fromCell], move.resources)

  def clone(self):
    return Board(self.radius, deepcopy(self.cells))

  def equals(self, anotherBoard):
    """

    :type anotherBoard: Board
    :return:
    """
    for c in self.cells.values():
      if c.id not in anotherBoard.cells:
        return False
      if c.resources != anotherBoard.cells[c.id].resources:
        return False
      if c.owner != anotherBoard.cells[c.id].owner:
        return False
    return True
