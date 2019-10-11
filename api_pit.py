from __future__ import unicode_literals
from flask import Flask, jsonify, request, send_from_directory
import time
from hexagon_gaming import *
import jsonpickle
import json
import threading
import game_runner
import api_player
import hexagon_agent


slots = {}
t = time.time()
app = Flask(__name__)
application = app
ui_assets_path = 'ui'


class Signaller:
  def __init__(self):
    self.signalled = False

  def signal(self):
    self.signalled = True

class SlotRunner:
  def __init__(self, slot):
    self.slot = slot
    self.runner = None
    self.running = False
    self.th = None
    self.sig = Signaller()

  def run(self, n_games=20, n_rounds=1000, radius=None):
    gr = game_runner.GameRunner(self.slot)

    def run_it():
      self.running = True
      gr.run_many(n_games, n_rounds, radius, self.sig)
      self.running = False
      self.th = None

    th = threading.Thread(target=run_it)
    th.setDaemon(True)
    self.th = th
    th.start()
    self.runner = gr

  def stop(self):
    if self.th is not None:
      self.sig.signal()
      self.th.join(0.5)
      self.th = None

@app.route('/', methods=['GET'])
def browse_default():
  try:
    return send_from_directory(ui_assets_path, 'index.html')
  except Exception as e:
    print(e.message)
    return e.message

@app.route('/<path:path>', methods=['GET'])
def statix(path):
   return send_from_directory(ui_assets_path, path)

@app.route('/api/slot/<slotName>', methods=['GET'])
def get_game_status(slotName):
  """

  :type slot: str
  :return:
  """
  global slots
  if slotName in slots and slots[slotName].runner is not None:
    ss = slots[slotName]
    game = ss.runner.game
    snapshot = GameSnapshot(game, ss.slot)
    return jsonify(json.loads(jsonpickle.dumps(snapshot)))
  else:
    return jsonify("game not valid"), 404

@app.route('/api/slot/<slotName>', methods=['DELETE'])
def stop_game(slotName):
  if slotName in slots:
    s = slots[slotName]
    s.stop()
    return jsonify(True), 204
  else:
    return jsonify("game not valid"), 404

@app.route('/api/slot/<slotName>', methods=['PUT'])
def create_game(slotName):
  j = request.json
  stop_game(slotName)
  n_games = 20
  n_rounds = 1000
  board_radius = None
  if request.args.get('n_games'):
    n_games = int(request.args.get('n_games'))
  if request.args.get('n_rounds'):
    n_rounds = int(request.args.get('n_rounds'))
  if request.args.get('radius'):
    board_radius = int(request.args.get('radius'))
  s = Slot(slotName, build_players(j))
  ss = slots[slotName] = SlotRunner(s)
  ss.run(n_games, n_rounds, board_radius)
  return jsonify(True), 201

def build_players(j):
  players = []
  for p in j['players']:
    name = p['name']
    if 'url' in p:
      players.append(api_player.ApiPlayer(name, p['url']))
    else:
      players.append(hexagon_agent.Aliostad(name))
  return players

def run_app():
  app.run(debug=False, use_reloader=False, host='0.0.0.0', port=19690, threaded=True)

if __name__ == '__main__':
  run_app()