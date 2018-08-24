import unittest
from short_memory import *


class ShortMemoryTests(unittest.TestCase):

  def test_can_append_but_not_too_much(self):
    m = ShortMemory(10)
    for i in range(0, 20):
      m.append(i)
    self.assertEqual(10, len(m))
    self.assertEqual(19, m[-1])

  def test_can_get_array_like(self):
    m = ShortMemory(10)
    for i in range(0, 20):
      m.append(i)
    a = m[5:]

    self.assertEqual(15, a[0])
    self.assertEqual(5, len(a))
