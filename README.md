# hexagon-rl
A Reinforcement Learning implementation for Danske Bank's Hexagon game. \

**IMPORTANT NOTE**: While the code is distributed under MIT license, **all legal rights for the Hexagon game itself belongs to [Danske Bank](https://danskebank.com/).** The game has been reproduced with permission and this reproduction is only intended for non-commercial and educational purposes.

![hexagon](https://raw.githubusercontent.com/aliostad/hexagon-rl/master/hexagon.jpg)

## What is Hexagon Game?
Hexagon is a simple round-based strategy game developed by [Danske Bank](https://danskebank.com/) ideal for autonomous agents. The game is played by two or more players/agents in a hexagonal shaped board made of hexagonal cells - similar to a bee hive - and the goal of the game is to occupy all cells within the game.

### Rules
Here is a brief summary of the rules but for full details, see [here](https://playhexagon.com/PublicRules):

 - The board is hexagonal shaped made of many hexagonal cells
 - At the start, each player starts with a single cell containing 100 resources. All the other cells will contain 0 resources (neutral cells).
 - A player at each round can play a single move:
   - **Capture**: A cell can capture a *neighbouring* neutral or enemy cell by moving some of its resources to that cell. The value of the cell after capture will be amount moved minus its resource value at the time of capture. For example a cell with 100 resources can capture a neighbouring cell containing 10 resources by moving 50 resources. After this move the captured cell will have 40 resources (50-10). Needless to say, it is illegal to try capturing a cell with moving resources less than the resource value of that cell.
   - **Boost**: A cell can accept any amount of resources from any other owned cell. It is a common strategy to boost cells bordering the enemy by moving resources from the centre.
 - At the end of each round, every owned cell (not neutral cells) will get an additional resource unless it has 100 or more resources.
 - If a player tries to play an illegal move (e.g. capturing a non-neighbouring cell), the move will be ignored.
 
### Implementation-related details
Danske Bank implementation has these characteristics:

 - Game is bounded by time and at the end, 
 - Agents are removed if they throw exceptions beyond a threshold
 - Each agent has at most 300ms for each move
 - The edges of the board fold on themselves. This means for example that the top and bottom page will be neighbouring ensuring that each cell even at the edges has 6 neghbours.
 - Before each move, the player receives information only on its owned cells and their neighbours.
 - Cells are named randomly.
 - Moves for cells are played in order hence moves for some players can overwrite other players (Being last player is better than being first player)

The implementation in this reproduction has these differences:
 - All edge cells (other than poles) have 7 neighbours rather than 6
 - Cells have consistent (according to hexagonal addressing) naming rather than random names
 - Order of plaing moves at each round is random hence likelihood of moves overwriting other moves is equal for all players.



