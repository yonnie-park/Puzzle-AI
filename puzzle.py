'''
Artificial Intelligence
Fall 2021
author: Jessie Park
uni: jp3645
'''
from __future__ import division
from __future__ import print_function

import sys
import math
import time
import queue as Q
from collections import deque
import resource
import heapq


#### SKELETON CODE ####
## The Class that Represents the Puzzle
class PuzzleState(object):
    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n*n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n*n)):
            raise Exception("Config contains invalid/duplicate entries : ", config)

        self.n         = n
        self.cost      = cost
        self.parent    = parent
        self.action    = action
        self.config    = config
        self.configstr = ''.join(str(i) for i in self.config)
        self.children  = []

        # Get the index and (row, col) of empty block
        self.blank_index = self.config.index(0)

    def display(self):
        """ Display this Puzzle state as a n*n board """
        for i in range(self.n):
            print(self.config[3*i : 3*(i+1)])

    def move_up(self):
        """ 
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """
        config = self.config[:]
    
        while self.blank_index not in [0,1,2]:
            config[self.blank_index] = config[self.blank_index-3]
            config[self.blank_index-3] = 0
            return PuzzleState(config, 3, parent=self, action="Up", cost=self.cost+1)

        return None
        
      
    def move_down(self):
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        config = self.config[:]

        while self.blank_index not in [6,7,8]:
            config[self.blank_index] = config[self.blank_index+3]
            config[self.blank_index+3] = 0
            return PuzzleState(config, 3, parent=self, action="Down", cost=self.cost+1)
            
        return None
      
    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        config = self.config[:]

        while self.blank_index not in [0,3,6]:
            config[self.blank_index] = config[self.blank_index-1]
            config[self.blank_index-1] = 0
            return PuzzleState(config, 3, parent=self, action="Left", cost=self.cost+1)
            
        return None

    def move_right(self):
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        config = self.config[:]

        while (self.blank_index) not in [2,5,8]:
            config[self.blank_index] = config[self.blank_index+1]
            config[self.blank_index+1] = 0
            return PuzzleState(config, 3, parent=self, action="Right", cost=self.cost+1)
            
        return None
      
    def expand(self):
        """ Generate the child nodes of this node """
        
        # Node has already been expanded
        if len(self.children) != 0:
            return self.children
        
        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        return self.children

# Function that Writes to output.txt

### Students need to change the method to have the corresponding parameters
def writeOutput(state, nodes_expanded, max_search_depth, running_time):
    ### Student Code Goes here

    path_to_goal = []
    
    cost_of_path = state.cost
    search_depth = cost_of_path

    while (state.action != "Initial"):
        path_to_goal.append(state.action)
        state = state.parent

    path_to_goal = path_to_goal[::-1]
    result=["path_to_goal: ", "cost_of_path: ", "nodes_expanded: ", "search_depth: ", "max_search_depth: "]
    
    f = open("output.txt", "w")
    f.write(result[0] + str(path_to_goal)+"\n")
    f.write(result[1]+ str(cost_of_path)+"\n")
    f.write(result[2] + str(nodes_expanded)+"\n")
    f.write(result[3]+ str(search_depth)+"\n")
    f.write(result[4] + str(max_search_depth)+"\n")
    f.write("running_time: %.8f" %running_time + "\n")
    f.write("max_ram_usage: %.8f"%float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1023E3)) 
    f.close()

def bfs_search(initial_state):
    """BFS search"""
    ### STUDENT CODE GOES HERE ###
    frontier = deque()
    frontier.append(initial_state)
    explored = set()
    explored.add(initial_state.configstr)
    
    max_search_depth = 0
    nodes_expanded = 0
    
    start_time=time.time()
    
    while frontier:
        state = frontier.popleft()

        if test_goal(state):
            end_time=time.time()
            running_time=end_time-start_time
            writeOutput(state, nodes_expanded, max_search_depth, running_time)
            return True

        nodes_expanded = nodes_expanded+1
        neighbors = state.expand()
        for neighbor in neighbors:
            if neighbor.configstr not in explored:
                if neighbor not in frontier:
                    max_search_depth = max(neighbor.cost, max_search_depth)
                    frontier.append(neighbor)
                    explored.add(neighbor.configstr)

                    
    return False


def dfs_search(initial_state):
    """DFS search"""
    ### STUDENT CODE GOES HERE ###
    start_time = time.time()
    frontier = [initial_state]
    explored = set()
    explored.add(initial_state.configstr)
    max_search_depth = 0
    nodes_expanded = 0
        
    while frontier:
        state = frontier.pop()
            
        if test_goal(state):
            end_time = time.time()
            running_time=end_time-start_time
            writeOutput(state, nodes_expanded, max_search_depth, running_time)
            return True
    
            nodes_expanded = nodes_expanded+1
            
        neighbors = state.expand()[::-1]
        for neighbor in neighbors:
            if neighbor.configstr not in explored:
                if neighbor not in frontier:
                    max_search_depth = max(neighbor.cost, max_search_depth)
                    frontier.append(neighbor)
                    explored.add(neighbor.configstr)


    return False
  

def A_star_search(initial_state):
    """A * search"""
    ### STUDENT CODE GOES HERE ###
    start_time = time.time()
    frontier = []
    explored = set()
    explored.add(initial_state.configstr)
    heapq.heappush(frontier, (0, time.time(), initial_state))
    max_search_depth = 0
    nodes_expanded = 0

    while frontier:
        state = heapq.heappop(frontier)
        board = state[2]
        max_search_depth = max(board.cost, max_search_depth)

        if test_goal(board):
            end_time = time.time()
            running_time=end_time-start_time
            writeOutput(board, nodes_expanded, max_search_depth, running_time)
            return True
        
        nodes_expanded = nodes_expanded+1
        for neighbor in board.expand():
            if neighbor.configstr not in explored:
                if neighbor not in frontier:
                    dist = calculate_total_cost(neighbor)
                    heapq.heappush(frontier, (time.time(), dist, neighbor))
                    explored.add(neighbor.configstr)

    return False


def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    ### STUDENT CODE GOES HERE ###
    cost = 0
    for idx in range(len(state.config)):
        value = state.config[idx]
        cost += state.cost + calculate_manhattan_dist(idx, value, 3)
    return cost

def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile"""
    ### STUDENT CODE GOES HERE ###
    x = (idx%3) - (value%n)
    y = (idx//3) - (value//3)
    dist = abs(x) + abs(y)
    return dist

def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    ### STUDENT CODE GOES HERE ###
    return puzzle_state.config == [0,1,2,3,4,5,6,7,8]

# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    search_mode = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = list(map(int, begin_state))
    board_size  = int(math.sqrt(len(begin_state)))
    hard_state  = PuzzleState(begin_state, board_size)
    start_time  = time.time()
    

    if   search_mode == "bfs": bfs_search(hard_state)
    elif search_mode == "dfs": dfs_search(hard_state)
    elif search_mode == "ast": A_star_search(hard_state)
    else: 
        print("Enter valid command arguments !")

    end_time = time.time()
    
    print("Program completed in %.3f second(s)"%(end_time-start_time))

if __name__ == '__main__':
    main()