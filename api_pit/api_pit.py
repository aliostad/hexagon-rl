from __future__ import unicode_literals
from flask import Flask, jsonify, request, send_from_directory
import time
import threading
from hexagon_gaming import *
import jsonpickle
import logging
import json

slots = {}
t = time.time()
app = Flask(__name__)
application = app
ui_assets_path = '../ui'

DEFAULT_SLOT = '1'


class SlotState:
  def __init__(self, slot):
    '''

    :type slot: Slot
    '''
    self.slot = slot
    self.th = None

  def start_game(self):
    pass

  def finish_game(self):
    pass

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
  if slotName in slots:
    game = slots[slotName]
    snapshot = GameSnapshot(game, slotName)
    return jsonify(json.loads(jsonpickle.dumps(snapshot)))

  else:
    return jsonify("game not valid"), 404

@app.route('/api/slot/<slotName>', methods=['PUT'])
def create_game(slotName):
  j = request.json

  slots[slotName]

def run_app():
  app.run(debug=False, use_reloader=False, host='0.0.0.0', port=19690, threaded=True)

if __name__ == '__main__':
  run_app()