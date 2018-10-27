from unittest import TestCase
from hexagon import *
from hexagon_gaming import Game
from hexagon_agent import Aliostad


class OtherTests(TestCase):

  def test_deep_copy_borad(self):
    b = Board(3)
    b2 = b.clone()
    cid = CellId(0, 0)
    self.assertNotEqual(b.cells[cid], b2.cells[cid])

  def test_deep_copy_game(self):
    g = Game("1", Aliostad("1"), Aliostad("-1"))
    g2 = g.clone()
    self.assertNotEqual(g.players[0], g2.players[0])