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

  def __str__(self):
    return '{}_{}'.format(self.nwes, self.x)

  def __hash__(self):
    return self.__str__().__hash__()

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
      return False, 'Cannot transfer {} from {} to {}. It is more than it has ()'.format(
        transfer, fromCell.id, self.id, fromCell.resources)
    if self.owner != fromCell.owner:
      if self.resources > transfer:
        return False, 'Cell {} cannot be captured since it has {} but transfer from {} is only {}'.format(
          self.id, self.resources, fromCell.id, transfer)
      elif not self.id in fromCell.id.get_neighbours():
        return False, 'Cell {} cannot capture {} since they are not neighbours'.format(fromCell.id, self.id)

    if self.owner == fromCell.owner:
      self.resources += transfer
    else:
      self.resources = transfer - self.resources
    fromCell.resources -= transfer
    return True, ''

  def to_neighbour_info(self, owner):
    NeighbourInfo(self.id, self.resources, None if self.owner == Cell.NoOwner else owner == self.owner)

  def increment_resources(self):
    self.resources = min(Cell.MaximumResource, self.resources+1)


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

  def __init__(self, radius):
    """
    :type radius: int
    """
    self.radius = radius
    self.diameter = radius*2 - 1
    self.ids = Board.build_ids_for_radius(radius)
    self.cells = {}

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

