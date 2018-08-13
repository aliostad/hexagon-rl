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
import unittest
from square_grid import *
from hexagon import *



class GridTests(unittest.TestCase):

  def test_conversion_from_grid_to_hex_correct_1_1(self):
    # 1, 1
    gid = GridCellId(1, 1)
    hid = gid.to_cell_id()
    self.assertEquals(1, hid.nwes)
    self.assertEquals(1, hid.x)

  def test_conversion_from_grid_to_hex_correct_3_3(self):
    # 3, 3
    gid = GridCellId(3, 3)
    hid = gid.to_cell_id()
    self.assertEquals(2, hid.nwes)
    self.assertEquals(3, hid.x)

  def test_conversion_from_grid_to_hex_correct_2_2(self):
    # 2, 2
    gid = GridCellId(2, 2)
    hid = gid.to_cell_id()
    self.assertEquals(1, hid.nwes)
    self.assertEquals(2, hid.x)

  def test_conversion_from_grid_to_hex_correct_3__1(self):
    # 3, -1
    gid = GridCellId(3, -1)
    hid = gid.to_cell_id()
    self.assertEquals(4, hid.nwes)
    self.assertEquals(-1, hid.x)

  def test_conversion_from_hex_to_grid_correct_1_1(self):

    # 1, 1
    hid = CellId(1, 1)
    gid = GridCellId.fromHexCellId(hid)
    self.assertEquals(1, gid.x)
    self.assertEquals(1, gid.y)

  def test_conversion_from_hex_to_grid_correct_3_2(self):

    # 3, 2
    hid = CellId(2, 3)
    gid = GridCellId.fromHexCellId(hid)
    self.assertEquals(3, gid.x)
    self.assertEquals(3, gid.y)

  def test_conversion_from_hex_to_grid_correct_2_1(self):
    # 2, 1
    hid = CellId(1, 2)
    gid = GridCellId.fromHexCellId(hid)
    self.assertEquals(2, gid.x)
    self.assertEquals(2, gid.y)

  def test_conversion_from_hex_to_grid_correct_2__1(self):
    # 2, -1
    hid = CellId(2, -1)
    gid = GridCellId.fromHexCellId(hid)
    self.assertEquals(1, gid.x)
    self.assertEquals(-1, gid.y)

