from __future__ import unicode_literals
from flask import Flask, jsonify, request, send_from_directory
import time
import threading
from hexagon_gaming import *
import jsonpickle
import logging
import json

games = {}
slot = None
t = time.time()
app = Flask(__name__)
application = app

class GameSnapshot:
  def __init__(self, game, slot):
    """

    :type game: Game
    """
    self.boardSnapshot = game.board.get_snapshot()
    self.stat = GameStat(game.name, game.round_no, game.get_player_stats(), False)
    self.radius = game.board.radius
    self.slot = slot

@app.route('/', methods=['GET'])
def browse_default():
  try:
    return send_from_directory('ui', 'index.html')
  except Exception as e:
    print(e.message)
    return e.message

@app.route('/<path:path>', methods=['GET'])
def staticx(path):
   return send_from_directory('ui', path)

@app.route('/api/game/<slotName>', methods=['GET'])
def get_game_status(slotName):
  """

  :type slot: str
  :return:
  """
  global games
  global slot
  if slotName in games:
    game = games[slotName]
    snapshot = GameSnapshot(game, slot)
    return jsonify(json.loads(jsonpickle.dumps(snapshot)))

  else:
    return jsonify("game not valid"), 404

def run_app():
  app.run(debug=False, use_reloader=False, host='0.0.0.0', port=19690, threaded=True)

class UiRunner:

  def __enter__(self):
    th = threading.Thread(target=run_app)
    th.setDaemon(True)
    th.start()
    self.thread = th

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.thread.join(1)

def run_in_background():
  log = logging.getLogger('werkzeug')
  log.disabled = True
  app.logger.disabled = True
  th = threading.Thread(target=run_app)
  th.setDaemon(True)
  th.start()
