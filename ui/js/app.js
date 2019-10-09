// some of the code from https://www.visualcinnamon.com/2013/07/self-organizing-maps-creating-hexagonal.html

$(function() {

var state = {
  baseUrl: "/api/slot/",
  slotName: "1",
  gameRunning: false,
  displayLoaded: false,
  updateInterval: 500
};

var qs = new URLSearchParams(location.search);
var slot = qs.get('slot');
if (slot)
    state.slotName = slot;

state.getUrl = function() {
  return state.baseUrl + state.slotName;
};

function getRandomColor() {
  var letters = '3456789ABC';
  var color = '#';
  for (var i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 10)];
  }
  return color;
}

var colours = {};

function getColour(playerName) {
  if(!colours[playerName])
    colours[playerName] = playerName ? getRandomColor() : "#CCC";
  return colours[playerName];
}

//
function getRowIndex(cellId, radius){
  return radius -1 + cellId.nwes;
}

function getColumnIndex(cellId, radius){
  var r = radius -1;
  var from = Math.max(-cellId.nwes-r, -r);
  return cellId.x - from;
}

function numberOfCellsInARow(nwes, radius){
  return (radius*2 -1) - Math.abs(nwes);
}

function buildPoints(radius) {
  var list = [];
  var r = radius-1;
  for (var i=-r; i<=r;i++){
    var from = Math.max(-i-r, -r);
    var to = Math.min(r-i, r);
    for (var j=from; j<=to; j++){
      list.push({nwes: i, x: j});
    }
  }
  return list;
}

function cellIdToCellRowColumn(cell, radius){
  var cellId = cell.id;
  var rowIndex = getRowIndex(cellId, radius);
  var columnIndex = getColumnIndex(cellId, radius);
  var numberOfCellsInTheRow = numberOfCellsInARow(cellId.nwes, radius);
  var maxPossible = radius*2 - 1;
  var r = rowIndex + 1;
  var q =  Math.round((maxPossible - numberOfCellsInTheRow)/2) + (columnIndex) + 1 + ((rowIndex* (radius-1)) % 2); // ((rowIndex* (radius-1)) % 2) => hacking code

  return {r: r, q: q, x: cellId.x, nwes: cellId.nwes,
    owner: cell.owner, resource: cell.resourceCount};
}

function draw(radius, pointsAxial, playerStats) {

  //svg sizes and margins
  var margin = {
      top: 10,
      right: 10,
      bottom: 10,
      left: 10
  },
  width = 750,
  height = 700;

  var displayMargin = 30;
  var statusDisplayLocation = {
    top: margin.top,
    left: margin.left + displayMargin + width,
    width: 200,
    height: height
  }
  var MapColumns = radius*2,
      MapRows = radius*2;

  //The maximum radius the hexagons can have to still fit the screen
  var hexRadius = d3.min([width/((MapColumns + 0.5) * Math.sqrt(3)),
     height/((MapRows + 1/3) * 1.5)]);

  var fontSize = Math.round(hexRadius * 0.7); // px
  var padding = fontSize / 9;

  var points = []
  var centreCoords = {x: width/2, y: height/2};
  for (var i = 0; i < pointsAxial.length; i++) {
    points.push(cellIdToCellRowColumn(pointsAxial[i], radius))
  }

  //Create SVG element
  var svg = d3.select("#chart").append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  //Set the hexagon radius
  var hexbin = d3.hexbin()
              .radius(hexRadius);

  var hexData = hexbin(points);

  //Draw the hexagons
  var g = svg.append("g");

  g
      .selectAll(".hexagon")
      .data(hexData)
      .enter()
      .append("path")
      .attr("class", "hexagon")
      .attr("d", function (d) {
        return "M" + d.x + "," + d.y + hexbin.hexagon();
      })
      .attr("stroke", "white")
      .attr("stroke-width", Math.round(padding) + "px")
      .attr("id", function (d, i) {
        return "Cell_" + d[0].nwes + "_" + d[0].x;
      })
      .style("fill", function (d,i) {
        return getColour(d[0].owner);
      })
      .append("title")
        .text(function (d, i) {
            return d[0].nwes + "_" + d[0].x;
        });

  g
      .selectAll("text")
      .data(hexData)
      .enter()
      .append("text")
      .attr("font-family", "arial")
      .attr("font-size", fontSize + "px")
      .attr("transform", "translate(0," + (hexRadius/4) + ")")
      .attr("fill", "beige")
      .style("text-anchor", "middle")
      .attr("title", function (d, i) {
        return d[0].nwes + "_" + d[0].x;
      })
      .attr("x", function(d, i) {
        return d.x;
      })
      .attr("id", function (d, i) {
        return "Text_" + d[0].nwes + "_" + d[0].x ;
      })
      .attr("y", function(d, i) { return d.y;})
      .text(function(d, i) {
        return d[0].resource;
      });
}

function initialDraw(data) {
  //var radius = 8;
  draw(data.radius, data.boardSnapshot.cells, data.stat.playerStats);
}

function resourceCountDisplay(resourceCount) {
  return resourceCount > 999
    ? Math.round(resourceCount / 1000) + "K"
    : resourceCount;
}

function updateCell(cell){
  var id = cell.id.nwes + "_" + cell.id.x;
  d3.select("#Cell_" + id)
    .style("fill", getColour(cell.owner));
  d3.select("#Text_" + id)
    .text(resourceCountDisplay(cell.resourceCount ? cell.resourceCount : cell.resources));
}

function updateCells(cells) {
  for(var c in cells){
    updateCell(cells[c]);
  }

}

function updateDisplayBoard(snapshot) {

  playerStats = snapshot.stat.playerStats;

  d3.select("#slotName")
    .text(state.slotName);

  d3.select("#gameName")
    .text(snapshot.stat.name);

  state.gameRunning = snapshot.stat.finished;

  d3.select("#gameStatus")
    .text(snapshot.stat.finished ? "finished" : snapshot.stat.round);

  if(playerStats.length > 0 && state.displayLoaded == false) {

    state.displayLoaded = true;

    d3.select("#display")
      .selectAll("div")
      .data(playerStats)
      .enter()
      .append("div")
      .attr("class", "standing")
      .style("padding", "12px")
      .style("font-family", "arial")
      .style("font-size", "18px")
      .style("color", "white")
      .attr("id", function(d, i){
        return "standing_" + i;
      });
  }

  for(var i in playerStats){
    var stat = playerStats[i];
    var id = "#standing_" + i;
    d3.select(id)
      .text(
      `[${snapshot.slot.player_scores[stat.playerName]}] ${stat.playerName} - ${stat.cellsOwned} (${stat.totalResource ? stat.totalResource : stat.totalResources})`)
      .style("background-color",
               getColour(stat.playerName));

  }
}

function setTimer() {

  $.ajax(state.getUrl() , {
      dataType: "json",
      type: "GET",
      success: function(data) {
        if (!state.displayLoaded)
            initialDraw(data);
        updateCells(data.boardSnapshot.cells);
        updateDisplayBoard(data);

        if (!data.stat.finished)
          setTimeout(setTimer, state.updateInterval);
      },
      error: function(xqh, status, e) {
        console.log(e);
        setTimeout(setTimer, state.updateInterval);
      }
     });
}

d3.select("#newGame")
  .on("click", function() {
  createGame();
});

setTimer();

function createGame() {

  $.ajax(state.getUrl(), {
    //dataType: "json",
    type: "POST",
    success: function(){

      state.gameRunning = true;
      state.displayLoaded = false;
      //d3.select("#chart").remove();
      //d3.select("#display").remove();

      $.ajax(state.getUrl(), {
          dataType: "json",
          type: "GET",
          success: function(data) {
            setTimer();
            initialDraw(data);
          },
          error: function(xqh, status, e) {
            console.log(e);
          }
       });
     },
     error: function(xqh, status, e) {
          console.log(e);
     }

  });

}

});
