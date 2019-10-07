from __future__ import unicode_literals
from flask import Flask, jsonify, request, send_from_directory
import time
import hexagon_agent
t = time.time()
app = Flask(__name__)
application = app

players = {}


@app.route("/api/player/<playerId>/game/<gameId>", methods=['POST'])
def startGame(playerId, gameId):
  global players
  try:
    players[key] = hexagon_agent.Aliostad(playerId)
    return '', 200
  except Exception as e:
    return e.message


@app.route("/api/player/<playerId>/game/<gameId>", methods=['PUT'])
def move(playerId, gameId):
  global players
  key = playerId + "_" + gameId
  if key not in players:
    return "player not found in this game", 404
  j = request.get_json()
  s = hexagon_agent.Klass(j)
  mv = players[key].turn(s.ownedCells)
  if mv is None:
    print('idiot')
  return jsonify(mv.__dict__)

@app.route("/api/player/<playerId>/game/<gameId>", methods=['DELETE'])
def endGame(playerId, gameId):
  global players
  key = playerId + "_" + gameId
  if key not in players:
    return "player not found in this game", 404
  del players[key]
  return "player not found in this game", 404