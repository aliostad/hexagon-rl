"""

          -2  -1  0
	  -2    x - x - x - x - x - x
          | / | / | / | / | / |
    -1    x - x - x - x - x - x
          | \ | \ | \ | \ | \ |
     0    x - x - x - x - x - x
          | / | / | / | / | / |
     1    x - x - x - x - x - x

       -2  -1  0   1   2   3
    -2  x - x - x - x - x - x
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
	   0    00- 01- 02- 03- 04- 05
          | / | / | / | / | / |
     1    10- 11- 12- 13- 14- 15
          | \ | \ | \ | \ | \ |
     2   2-1- 20- 21- 22- 23- 24
          | / | / | / | / | / |
     3   3-1- 30- 31- 32- 33- 34

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
    return GridCellId(cellId.nwes + (cellId.x + (-1 if cellId.x < 1 else 0)) / 2, cellId.x)

  @staticmethod
  def sign(x):
    return -1 if x < 0 else 1

  def to_cell_id(self):
    return CellId(self.x - (self.y + (-1 if self.y < 1 else 0)) / 2, self.y)
