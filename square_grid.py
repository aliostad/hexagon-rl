"""

          -2  -1  0
	  -2    x - x - x - x - x - x
          | / | / | / | / | / |
    -1    x - x - x - x - x - x
          | \ | \ | \ | \ | \ |
    0     x - x - x - x - x - x
          | / | / | / | / | / |
    1     x - x - x - x - x - x

       -2  -1  0   1   2   3
    -2  x - x - x - x - x - xm
         \ / \ / \ / \ / \ / \
    -1    x - x - x - x - x - x
         / \ / \ / \ / \ / \ /
     0  x - x - x - x - x - x
         \ / \ / \ / \ / \ / \
     1    x - x - x - x - x - x


        0   1   2   3   4   5
     0  00- 01- 02- 03- 04- 05
         \ / \ / \ / \ / \ / \
     1    10- 11- 12- 13- 14- 15
         / \ / \ / \ / \ / \ /
     2 2-1- 20- 21- 22- 23- 24
         \ / \ / \ / \ / \ / \
     3   3-1- 30- 31- 32- 33- 34


     	    0   1   2   3   4   5
	   0    x - x - x - x - x - x
          | / | / | / | / | / |
     1    x - x - x - x - x - x
          | \ | \ | \ | \ | \ |
     2    x - x - x - x - x - x
          | / | / | / | / | / |
     3    x - x - x - x - x - x


     	    0   1   2   3   4   5

    -1   -11 -12 -13 -14 -15 -16
          | \ | \ | \ | \ | \ |
	   0    00- 01- 02- 03- 04- 05
          | / | / | / | / | / |
     1    10- 11- 12- 13- 14- 15
          | \ | \ | \ | \ | \ |
     2    2-1- 20- 21- 22- 23- 24
          | / | / | / | / | / |
     3    3-1- 30- 31- 32- 33- 34



"""
from hexagon import *
import math


class GridCellId():

  def __init__(self, x, y):
    """

    :type x: int
    :type y: int
    """
    self.x = x
    self.y = y

  @staticmethod
  def fromHexCellId(cellId):
    """

    :type CellId:
    :return:
    """
    return GridCellId(cellId.nwes + (cellId.x / 2), cellId.x)

  def to_cell_id(self):
    return CellId(self.x - (self.y/2), self.y)

  def transpose(self, x0, y0):
    """

    :type x0: int
    :type y0: int
    :return:
    """
    return GridCellId(self.x + x0, self.y + y0)

  def __repr__(self):
    return "{}_{}".format(self.x, self.y)


def get_thid_from_cellId(cid, rect_width):
  """

  :type cid: CellId
  :type rect_width: int
  :return:
  """
  hid = GridCellId.fromHexCellId(cid)
  thid = hid.transpose(rect_width / 2, rect_width / 2)
  return thid


def get_index_from_cellId(cid, rect_width, number_of_features=1):
  """

  :type cid: CellId
  :type rect_width: int
  :return:
  """
  thid = get_thid_from_cellId(cid, rect_width)
  idx = thid.y * rect_width + thid.x
  return idx * number_of_features


def get_cellId_from_hid(hid, rect_width):
  """

  :type index: GridCellId
  :type rect_width: int
  :return:
  """
  thid = hid.transpose(-(rect_width / 2), -(rect_width / 2))
  cellId = thid.to_cell_id()
  return cellId


def get_cellId_from_index(index, rect_width, number_of_features=1):
  """

  :type index: int
  :type rect_width: int
  :return:
  """
  index = index / number_of_features
  y = index / rect_width
  x = index % rect_width
  hid = GridCellId(x, y)
  return get_cellId_from_hid(hid, rect_width)

