import unittest
from hexagon import *


class CellTests(unittest.TestCase):

  def test_same_id_transfer(self):
    name = 'man'
    id = CellId(0, 0)
    c1 = Cell(id, name, 100, None)
    c2 = Cell(id, name, 100, None)
    is_success, error = c1.try_transfer_from(c2, 50)
    self.assertFalse(is_success)
    print(error)

  def test_can_capture(self):
    c1 = Cell(CellId(0, 0), 'suma', 100, None)
    c2 = Cell(CellId(1, -1), 'tropin', 10, None)
    is_success, error = c2.try_transfer_from(c1, 50)
    self.assertTrue(is_success)
    self.assertEqual(c1.resources, 50)
    self.assertEqual(c2.resources, 40)

  def test_can_boost(self):
    c1 = Cell(CellId(0, 0), 'suma', 100, None)
    c2 = Cell(CellId(1, -1), 'suma', 10, None)
    is_success, error = c2.try_transfer_from(c1, 50)
    self.assertTrue(is_success)
    self.assertEqual(c1.resources, 50)
    self.assertEqual(c2.resources, 60)

  def test_5_0_is_apole_for_radius_6(self):
    c = CellId(5, 0)
    self.assertTrue(c.is_pole(6))


class BoardTests(unittest.TestCase):

  def test_board_construct_node_count(self):
    radius = 2
    b = Board(radius)
    self.assertEqual(7, len(b.cells))

    self.assertEqual(1 + 6 + (2 * 6) + (3 * 6) + (4 * 6) + (5 * 6), len(Board(6).cells))

  def test_board_construct_node_neighbour_count(self):
    b = Board(6)
    for cid in b.cells:
      cnt = len(b.cells[cid].neighbours)
      self.assertGreater(cnt, 5)
      self.assertLess(cnt, 8)

  def test_get_cell_info(self):
    b = Board(6)
    centre = CellId(0, 0)
    cell = b.cells[centre]
    cell.owner = 'Ali'
    cell.resources = 100

    infos = b.get_cell_infos_for_player('Ali')
    self.assertEqual(1, len(infos))
    self.assertEqual(6, len(infos[0].neighbours))
    self.assertTrue(infos[0].neighbours[0].isOwned is None)
