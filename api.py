from __future__ import unicode_literals
from flask import Flask, jsonify, request, send_from_directory
import logging
from logging import FileHandler
import time
import numpy as np
import os
import codecs
import random
import sys
import hexagon

t = time.time()
app = Flask(__name__)
application = app

players = {}

@app.route("/api/player/<playerId>/game/<gameId>", methods=['POST'])
def startGame(playerId, gameId):
  try:
    key = playerId + "_" + gameId
    players[key] = hexagon.Aliostad(playerId)
    return '', 200
  except Exception as e:
    return e.message

@app.route("/api/player/<playerId>/game/<gameId>", methods=['PUT'])
def move(playerId, gameId):
  try:
    key = playerId + "_" + gameId
    if key not in players:
      return "player not found in this game", 404
    j = request.get_json()
    s = hexagon.Klass(j)
    mv = players[key].turn(s.ownedCells)
    return jsonify(mv.__dict__)
  except Exception as e:
    return e.message

@app.route("/api/player/<playerId>/game/<gameId>", methods=['DELETE'])
def endGame(playerId, gameId):
  key = playerId + "_" + gameId
  if key not in players:
    return "player not found in this game", 404
  del players[key]
  return "player not found in this game", 404