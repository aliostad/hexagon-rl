from hexagon_gaming import *
import requests
import urlparse
from . import MoveFeedback
import jsonpickle
import json

paths = {
  'join-game': 'game/{}',
  'finish-game': 'game/{}',
  'move': 'game/{}',
  'move-feedback': 'game/{}/move/{}/feedback',
}

class ApiPlayer(Player):
  def __init__(self, name, baseUrl):
    Player.__init__(self, name)
    self.url = baseUrl

  def move(self, playerView):
    '''

    :type playerView: PlayerView
    :return:
    '''
    relativ = paths['move'].format(self.current_game)
    url = urlparse.urljoin(self.url, relativ)
    try:
      r = requests.put(url, json.loads(jsonpickle.dumps(playerView)))
      if r.status_code == 200:
        j = json.loads(r.json())
        return Move(CellId.parse(j['fromCell']), CellId.parse(j['toCell']), j['resource'])
      else:
        print r.text
        return None
    except Exception as e:
      print e
      return None
  def start(self, gameName):
    '''

    :type gameName: str
    :return:
    '''
    relativ =  paths['join-game'].format(gameName)
    url = urlparse.urljoin(self.url, relativ)
    try:
      r = requests.post(url)
      if r.status_code == 204:
        self.current_game = gameName
        return True
      else:
        return False
    except Exception as e:
      print e
      return False

  def finish(self):
    if not self._started:
      raise RuntimeError('Player not started')
    try:
      relativ = paths['finish-game'].format(self.current_game)
      url = urlparse.urljoin(self.url, relativ)
      r = requests.post(url)
    except Exception as e:
      # not important frankly
      print e
    self._started = False

  def move_feedback(self, roundNo, move, error):
    relativ = paths['move-feedback'].format(self.current_game, roundNo)
    url = urlparse.urljoin(self.url, relativ)
    mf = MoveFeedback(move, error)
    try:
      r = requests.put(url, json.loads(jsonpickle.dumps(mf)))
    except Exception as e:
      # not important frankly
      print e

