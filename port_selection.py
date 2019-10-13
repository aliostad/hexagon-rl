import os

def Select_port(default_port, *args):
  '''

  :type default_port: int
  :param args:
  :return: port
  '''
  port = default_port
  for arg in args:
    if arg in os.environ:
      port = int(os.environ[arg])
  return port