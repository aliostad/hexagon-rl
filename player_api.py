'''
A REST API representing a player (or a number of)
'''
from __future__ import unicode_literals
from flask import Flask, jsonify, request, send_from_directory
import time
import hexagon_agent
import json
from hexagon_gaming import *

t = time.time()
app = Flask(__name__)
application = app

players = {}


@app.route("/api/player/<playerId>/game/<gameId>", methods=['POST'])
def startGame(playerId, gameId):
  global players
  try:
    players[playerId] = hexagon_agent.Aliostad(playerId)
    return '', 204
  except Exception as e:
    return e.message


@app.route("/api/player/<playerId>/game/<gameId>", methods=['PUT'])
def move(playerId, gameId):
  global players
  key = playerId
  if key not in players:
    return "player not found in this game", 404
  j = json.loads(request.json)
  playerview = create_player_view_from_j(j)
  mv = players[key].move(playerview)
  if mv is None:
    print('idiot')

  return jsonify(json.dumps({
    'toCell': str(mv.toCell),
    'resources': mv.resources,
    'fromCell': str(mv.fromCell)
  }))

@app.route("/api/player/<playerId>/game/<gameId>", methods=['DELETE'])
def endGame(playerId, gameId):
  global players
  key = playerId
  if key not in players:
    return "player not found in this game", 404
  del players[key]
  return "player not found in this game", 404

def create_player_view_from_j(j):
  rounNo = j['roundNo']
  ownedCells = []
  for oc in j['ownedCells']:
    ci = CellInfo(CellId.parse(oc['id']), oc['resources'], [])
    for n in oc['neighbours']:
      ni = NeighbourInfo(CellId.parse(n['id']), n['resources'], n['isOwned'])
      ci.neighbours.append(ni)
    ownedCells.append(ci)
  return PlayerView(rounNo, ownedCells)
