"""
A short sequential array-like memory
"""
class ShortMemory:

  def __init__(self, capacity):
    self.capacity = capacity
    self.store = []

  def __delitem__(self, index):
      del self.store[index]

  def __getitem__(self, index):
      return self.store[index]

  def __len__(self):
    return len(self.store)

  def append(self, value):
    """

    :param value: value to add to list
    :return: last removed item
    """
    self.store.append(value)
    ret = None
    if len(self.store) > self.capacity:
      ret = self.store[0]
      del self.store[0]
    return ret
