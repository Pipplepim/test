const fs = require("fs");
const path = require("path");
const os = require("os");

const writeToFile = (name, code) => {
  const desktopPath = path.join(os.homedir(), "Desktop");

  const aiFolderPath = path.join(desktopPath, "AI");

  if (!fs.existsSync(aiFolderPath)) {
    fs.mkdirSync(aiFolderPath);
  }

  const fileName = `${name}.py`;
  const filePath = path.join(aiFolderPath, fileName);

  fs.writeFileSync(filePath, code);

  console.log(`File written successfully: ${filePath}`);
};

const ai = {
  alphabeta: function alphabeta() {
    const code = `
def minimax(nodes, depth, alpha, beta, player, index=0):
  if depth == 0 or 2*index >= len(nodes): return nodes[index]

  if player == 1:
      maximum = float('-inf')
      maximum = max(maximum, minimax(nodes, depth-1, alpha, beta, False, 2*index))
      alpha = max(alpha, maximum)
      if beta <= alpha: return maximum
      maximum = max(maximum, minimax(nodes, depth-1, alpha, beta, False, 2*index+1))
      alpha = max(alpha, maximum)
      return maximum

  else:
      minimum = float('inf')
      minimum = min(minimum, minimax(nodes, depth-1, alpha, beta, True, 2*index))
      beta = min(beta, minimum)
      if beta <= alpha: return minimum
      minimum = min(minimum, minimax(nodes, depth-1, alpha, beta, True, 2*index+1))
      beta = min(beta, minimum)
      return minimum

def main():
  player = int(input("Player 1 or 2? "))
  depth = int(input("Enter depth of tree: "))
  nodes = [0 for _ in range(2 ** depth)]
  for i in range(len(nodes)):
      nodes[i] = int(input(f"Enter value of terminal node {i+1}: "))
  best = minimax(nodes, depth, float('-inf'), float('inf'), player)
  print(f"Best for Player {player}: {best}")

main()
`;
    writeToFile("alphabeta", code);
    return;
  },

  astar: function astar() {
    const code = `
from queue import PriorityQueue
from collections import defaultdict

def a_star(graph, start, heuristic):
  fringe = PriorityQueue()
  path = []
  distance = 0

  fringe.put((heuristic[start] + distance, [start, 0]))
  while fringe.empty() == False:
    [currentnode, travelled] = fringe.get()[1]
    path.append(currentnode)
    distance += travelled
    if currentnode == "B": break
    fringe = PriorityQueue()
    for [neighbor, gn] in graph[currentnode]:
      if neighbor not in path: fringe.put((heuristic[neighbor] + gn + distance, [neighbor, gn]))
  return path

def add(graph, u, v, cost):
  graph[u].append((v, cost))
  graph[v].append((u, cost))

def main():
  graph = defaultdict(list)
  add(graph, "A", "Z", 75)
  add(graph, "A", "T", 118)
  add(graph, "A", "S", 140)
  add(graph, "Z", "O", 71)
  add(graph, "O", "S", 151)
  add(graph, "T", "L", 111)
  add(graph, "L", "M", 70)
  add(graph, "M", "D", 75)
  add(graph, "D", "C", 120)
  add(graph, "S", "R", 80)
  add(graph, "S", "F", 99)
  add(graph, "R", "C", 146)
  add(graph, "C", "P", 138)
  add(graph, "R", "P", 97)
  add(graph, "F", "B", 211)
  add(graph, "P", "B", 101)
  add(graph, "B", "G", 90)
  add(graph, "B", "U", 85)
  add(graph, "U", "H", 98)
  add(graph, "H", "E", 86)
  add(graph, "U", "V", 142)
  add(graph, "V", "I", 92)
  add(graph, "I", "N", 87)

  heuristic = {
    "A":  366, "B": 0, "C": 160, "D": 242, "E": 161, "F": 178, "G": 77,
    "H": 151, "I": 226, "L": 244, "M": 241, "N": 234, "O": 380, "P": 98,
    "R": 193, "S": 253, "T": 329, "U": 80, "V": 199, "Z": 374
  }

  start = input("Enter start node: ")
  path = a_star(graph, start, heuristic)
  if path: print(path)
  else: print("No path found")

main()
`;
    writeToFile("astar", code);
    return;
  },

  hillclimbing: function hillclimbing() {
    const code = `
import random

def f(x): return -(x - 3)**2 + 10

def hillclimbing(start):
  optima = start
  height = f(start)
  while True:
    leftNeighbor = optima - 0.01
    rightNeighbor = optima + 0.01
    leftHeight = f(leftNeighbor)
    rightHeight = f(rightNeighbor)
    if leftHeight >= height and rightHeight >= height:
      if leftHeight >= rightHeight:
        optima = leftNeighbor
        height = leftHeight
      else:
        optima = rightNeighbor
        height = rightHeight
    elif leftHeight >= height:
      optima = leftNeighbor
      height = leftHeight
    elif rightHeight >= height:
      optima = rightNeighbor
      height = rightHeight
    else: break
  return optima, height

def main():
  start = round(random.uniform(0, 10), 3)
  optima, height = hillclimbing(start)
  print(f"Starting Point: {start}")
  print(f"Local Optima: {optima}")
  print(f"f(x): {height}")

main()
`;
    writeToFile("hillclimbing", code);
    return;
  },

  idfs: function idfs() {
    const code = `
from collections import defaultdict

def dls(graph, currentnode, goal, limit, path):
  if limit == 0:
    if currentnode == goal: return path + [currentnode]
    else: return None
  elif limit > 0:
    path = path + [currentnode]
    if currentnode == goal: return path
    for neighbor, cost in graph[currentnode]:
      if neighbor not in path:
        result = dls(graph, neighbor, goal, limit - 1, path)
        if result is not None: return result
    return None

def idfs(graph, start, goal):
  depth = 0
  while True:
    result = dls(graph, start, goal, depth, [])
    if result is not None: return result
    depth += 1

def add(graph, u, v, cost):
  graph[u].append((v, cost))
  graph[v].append((u, cost))

def main():
  graph = defaultdict(list)
  add(graph, "A", "Z", 75)
  add(graph, "A", "T", 118)
  add(graph, "A", "S", 140)
  add(graph, "Z", "O", 71)
  add(graph, "O", "S", 151)
  add(graph, "T", "L", 111)
  add(graph, "L", "M", 70)
  add(graph, "M", "D", 75)
  add(graph, "D", "C", 120)
  add(graph, "S", "R", 80)
  add(graph, "S", "F", 99)
  add(graph, "R", "C", 146)
  add(graph, "C", "P", 138)
  add(graph, "R", "P", 97)
  add(graph, "F", "B", 211)
  add(graph, "P", "B", 101)
  add(graph, "B", "G", 90)
  add(graph, "B", "U", 85)
  add(graph, "U", "H", 98)
  add(graph, "H", "E", 86)
  add(graph, "U", "V", 142)
  add(graph, "V", "I", 92)
  add(graph, "I", "N", 87)

  start = input("Enter start node: ")
  goal = "B"
  path = idfs(graph, start, goal)
  if path: print(f"Path from {start} to {goal}: {path}")
  else: print("No path found")

if __name__ == '__main__': main()
`;
    writeToFile("idfs", code);
    return;
  },

  nqueens: function nqueens() {
    const code = `
      
def solve_n_queens(n):
  def backtrack(row=0, diagonals=set(), anti_diagonals=set(), cols=set(), board=[]):
    if row == n:
      result.append(["".join(row) for row in board])
      return
    
    for col in range(n):
      diag = row - col
      anti_diag = row + col
      if col in cols or diag in diagonals or anti_diag in anti_diagonals:
          continue

      cols.add(col)
      diagonals.add(diag)
      anti_diagonals.add(anti_diag)
      board.append("." * col + "Q" + "." * (n - col - 1))
      
      backtrack(row + 1, diagonals, anti_diagonals, cols, board)
      
      cols.remove(col)
      diagonals.remove(diag)
      anti_diagonals.remove(anti_diag)
      board.pop()

  result = []
  backtrack()
  return result

n = 4
solutions = solve_n_queens(n)
for sol in solutions:
    for row in sol:
        print(row)
    print()
print(len(solutions))
`;
    writeToFile("nqueens", code);
    return;
  },

  rbfs: function rbfs() {
    const code = `
from collections import defaultdict

def rbfs(graph, node, f_limit, g_cost, heuristic):
  if node == "B": return [node], g_cost

  successors = []
  for (neighbor, cost) in graph[node]:
    g = g_cost + cost
    f = g + heuristic[neighbor]
    successors.append([neighbor, g, f])

  if len(successors) == 0: return None, float('inf')

  while True:
    successors.sort(key=lambda x: x[2]) # sort by f - 3rd value
    best = successors[0]
    if best[2] > f_limit: return None, best[2]

    if len(successors) > 1: alternative = successors[1][2]
    else: alternative = float('inf')

    result, best_f = rbfs(graph, best[0], min(f_limit, alternative), best[1], heuristic)
    best[2] = best_f
    if result is not None: return [node] + result, best_f
    if best[2] > f_limit: return None, best[2]

def add(graph, u, v, cost):
  graph[u].append((v, cost))
  graph[v].append((u, cost))

def main():
  graph = defaultdict(list)
  add(graph, "A", "Z", 75)
  add(graph, "A", "T", 118)
  add(graph, "A", "S", 140)
  add(graph, "Z", "O", 71)
  add(graph, "O", "S", 151)
  add(graph, "T", "L", 111)
  add(graph, "L", "M", 70)
  add(graph, "M", "D", 75)
  add(graph, "D", "C", 120)
  add(graph, "S", "R", 80)
  add(graph, "S", "F", 99)
  add(graph, "R", "C", 146)
  add(graph, "C", "P", 138)
  add(graph, "R", "P", 97)
  add(graph, "F", "B", 211)
  add(graph, "P", "B", 101)
  add(graph, "B", "G", 90)
  add(graph, "B", "U", 85)
  add(graph, "U", "H", 98)
  add(graph, "H", "E", 86)
  add(graph, "U", "V", 142)
  add(graph, "V", "I", 92)
  add(graph, "I", "N", 87)

  heuristic = {
    "A":  366, "B": 0, "C": 160, "D": 242, "E": 161, "F": 178, "G": 77,
    "H": 151, "I": 226, "L": 244, "M": 241, "N": 234, "O": 380, "P": 98,
    "R": 193, "S": 253, "T": 329, "U": 80, "V": 199, "Z": 374
  }

  start = input("Enter start node: ")
  path, cost = rbfs(graph, start, float('inf'), 0, heuristic)
  if path:
    print(f"Path from {start} to B: {path}")
    print(f"Cost: {cost}")
  else: print("No path found")

if __name__ == '__main__': main()
`;
    writeToFile("rbfs", code);
    return;
  },

  waterjug: function waterjug() {
    const code = `
def pour(source, destination, current, volume1, volume2):
  jug1, jug2 = current
  if source == "J1":
    if jug1 > 0 and jug2 < volume2:
      amount = min(jug1, volume2 - jug2)
      updatedjug1 = jug1 - amount
      updatedjug2 = jug2 + amount
      return (updatedjug1, updatedjug2)
  else:
    if jug2 > 0 and jug1 < volume1:
      amount = min(jug2, volume1 - jug1)
      updatedjug1 = jug1 + amount
      updatedjug2 = jug2 - amount
      return (updatedjug1, updatedjug2)
  return None

def solve(volume1, volume2, target):
  seen = set()
  stack = [((0, 0), [])]
  while stack:
    current, moves = stack.pop()
    if current[0] == target or current[1] == target: return moves # final result
    seen.add(current)
    for move in ["FJ1", "EJ1", "FJ2", "EJ2", "PJ2J1", "PJ1J2"]:
      if move == "FJ1": new = (volume1, current[1])
      elif move == "FJ2": new = (current[0], volume2)
      elif move == "EJ1": new = (0, current[1])
      elif move == "EJ2": new = (current[0], 0)
      elif move == "PJ1J2": new = pour("J1", "J2", current, volume1, volume2)
      elif move == "PJ2J1": new = pour("J2", "J1", current, volume1, volume2)

      if new and new not in seen:
        updatedmoves = moves + [move]
        stack.append((new, updatedmoves))
      

  return None 

def main():
  volume1 = int(input("Enter volume of first jug: "))
  volume2 = int(input("Enter volume of second jug: "))
  target = int(input("Enter volume to be measured: "))
  solution = solve(volume1, volume2, target)
  if solution is None: print("No solution exists")
  else:
    for move in solution:
      text = ''
      if(move == "FJ1"): text = "Fill Jug 1"
      elif(move == "FJ2"): text = "Fill Jug 2"
      elif(move == "EJ1"): text = "Empty Jug 1"
      elif(move == "EJ2"): text = "Empty Jug 2"
      elif(move == "PJ1J2"): text = "Pour Jug 1 into Jug 2"
      elif(move == "PJ2J1"): text = "Pour Jug 2 into Jug 1"
      print(text)

main()
`;
    writeToFile("waterjug", code);
    return;
  },

  wumpus: function wumpup() {
    const code = `
import random

r_world = [
    [['0'], ['0'], ['0'], ['G']],
    [['0'], ['B'], ['B'], ['B']],
    [['0'], ['B', 'S'], ['S', 'P'], ['S']],
    [['0'], ['S', 'B'], ['W', 'B'], ['S', 'B']]
]

def write(i, j, data):
  allowed = directions(i, j)
  for k in allowed:
    if k not in visited and v_world[k[0]][k[1]] != '0':
      v_world[k[0]][k[1]].append(data)

def directions(i, j):
  allowed = []
  if i > 0:  
      allowed.append((i - 1, j))
  if i < len(r_world) - 1:  
      allowed.append((i + 1, j))
  if j > 0:  
      allowed.append((i, j - 1))
  if j < len(r_world[0]) - 1: 
      allowed.append((i, j + 1))
  return allowed

def kb(i, j, sense):
  if sense == 'S':
      predict = '?W'
  elif sense == 'B':
      predict = '?P'
  v_world[i][j].clear()
  v_world[i][j].append(sense)

  allowed = directions(i, j)
  print("Allowed Steps:", allowed)

  take_this_steps = []
  for k in allowed:
      if '0' in v_world[k[0]][k[1]] or len(v_world[k[0]][k[1]]) == 0:
          if k not in visited:
              take_this_steps.append(k)

  if len(take_this_steps) == 0:

      for k in range(len(visited)):
          if (i, j) == visited[k]:
              i, j = visited[k - 1][0], visited[k - 1][1]
              break
  else:

      rand = random.randint(0, len(take_this_steps) - 1)
      i, j = take_this_steps[rand][0], take_this_steps[rand][1]
      visited.append(take_this_steps[rand])

  return i, j

v_world = [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]]
visited = []

def wumpus(i, j):
    try:
        count = 0
        while 'G' not in r_world[i][j]:
            if count > 30:
                raise Exception("Inadequate Environment")
            if '0' in r_world[i][j]:
                i, j = kb(i, j, '0')
            else:
                if 'S' in r_world[i][j]:
                    write(i, j, '?W')
                    i, j = kb(i, j, 'S')
                if 'B' in r_world[i][j]:
                    write(i, j, '?P')
                    i, j = kb(i, j, 'B')

            print("Going To", i, j)
            count += 1
            for m in v_world:
                print(m)
            print()
        print("Gold Found at", i, j)
        reward = 1000 - len(visited) - 1
        print("Rewarded", reward)
        return
    except Exception as e:
        print(e)


v_world[3][0].append('0')
visited.append((3, 0))
wumpus(3, 0)
print("Visited Path:", visited)
    `;
    writeToFile("wumpus", code);
    return;
  },
};

module.exports = ai;
