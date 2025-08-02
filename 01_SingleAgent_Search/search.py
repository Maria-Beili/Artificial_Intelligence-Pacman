# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in search_agents.py).
"""
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in obj-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem.
        """
        util.raise_not_defined()

    def is_goal_state(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raise_not_defined()

    def get_successors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raise_not_defined()

    def get_cost_of_actions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raise_not_defined()


def tiny_maze_search(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# def addSuccessors(problem, addCost=True):

class SearchNode:
    def __init__(self, parent, node_info):
        """
            parent: parent SearchNode.

            node_info: tuple with three elements => (coord, action, cost)

            coord: (x,y) coordinates of the node position

            action: Direction of movement required to reach node from
            parent node. Possible values are defined by class Directions from
            game.py

            cost: cost of reaching this node from the starting node.
        """

        self.__state = node_info[0]
        self.action = node_info[1]
        self.cost = node_info[2] if parent is None else node_info[2] + parent.cost
        self.parent = parent

    # The coordinates of a node cannot be modified, se we just define a getter.
    # This allows the class to be hashable.
    @property
    def state(self):
        return self.__state

    def get_path(self):
        path = []
        current_node = self
        while current_node.parent is not None:
            path.append(current_node.action)
            current_node = current_node.parent
        path.reverse()
        return path
    
    # Consider 2 nodes to be equal if their coordinates are equal (regardless of everything else)
    # def __eq__(self, __o: obj) -> bool:
    #     if (type(__o) is SearchNode):
    #         return self.__state == __o.__state
    #     return False

    # # def __hash__(self) -> int:
    # #     return hash(self.__state)

def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state())
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    "*** YOUR CODE HERE ***"
    # Scheme LIFO
    # Initialize the stack to store the nodes to be explored
    stack = util.Stack()

    # Initialize the stack with the start state and the path to reach it (empty)
    stack.push(SearchNode(None, (problem.get_start_state(), None, 0)))

    # Initialize the explored set that will store the states that have been visited
    set_visited = set()
    
    # We will explore the nodes until the stack is empty
    while not stack.is_empty():

        # Get the current node from the stack
        current_node = stack.pop()

        # Case 1: Check if the current node is the goal state and return the path if it is
        if problem.is_goal_state(current_node.state):
            return current_node.get_path()
        
        # Case 2: Check if the current node has been visited before. If not, add it to the explored set
        if current_node.state not in set_visited:
            set_visited.add(current_node.state)

            # Add the successors of the current node to the stack if they have not been visited before
            for successor in problem.get_successors(current_node.state):
                if successor[0] not in set_visited:
                    stack.push(SearchNode(current_node, successor))
                    
    # If the goal state is not found, return an empty list
    return []
    # util.raise_not_defined()

    '''
    Question 1 (3 points): Finding a Fixed Food Dot using Depth First Search

        This DFS implementation explores the deepest nodes in the search tree first. 
        It maintains the necessary information to reconstruct the path from the start state to the goal state using a search node class. 
        Each node contains the current state, the action taken to reach it, and the cost of reaching it from the start state. 
        The algorithm utilizes a stack to manage the frontier, adding successors of each node until the goal state is found or all nodes are explored. 
        It returns a list of valid actions to move Pacman from the start state to the goal state, not considering the cost neither the efficiency of the path.

    '''


def breadth_first_search(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # scheme FIFO
    # Initialize the queue to store the nodes to be explored
    queue = util.Queue()

    # Initialize the queue with the start state and the path to reach it (empty)
    queue.push(SearchNode(None, (problem.get_start_state(), None, 0)))

    # Initialize the explored set that will store the states that have been visited
    set_visited = set()

    # We will explore the nodes until the queue is empty
    while not queue.is_empty():
        # Get the current node from the queue
        current_node = queue.pop()

        # Case 1: Check if the current node is the goal state and return the path if it is
        if problem.is_goal_state(current_node.state):
            return current_node.get_path()
        
        # Case 2: Check if the current node has been visited before. If not, add it to the explored set
        if current_node.state not in set_visited:
            set_visited.add(current_node.state)

            # Add the successors of the current node to the queue if they have not been visited before
            for successor in problem.get_successors(current_node.state):
                queue.push(SearchNode(current_node, successor))

    # If the goal state is not found, return an empty list            
    return []
    # util.raise_not_defined()


    ''' 
    Question 2 (3 points): Breadth First Search

        This BFS implementation explores the shallowest nodes first. It uses a queue for the frontier and tracks visited states. 
        Each node stores the current state, action, and cost. 
        The algorithm finds the goal by expanding nodes per level, which ensures the shortest path in a maze with uniform costs.
        Returns the sequence of actions leading Pacman from the start to the goal state.

    '''


def uniform_cost_search(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"    
    # Initialize the priority queue to store the nodes to be explored
    in_queue = util.PriorityQueue()

    # Initialize the queue with the start state and the path to reach it (empty)
    in_queue.push(SearchNode(None, (problem.get_start_state(), None, 0)), 0)

    # Initialize the explored set that will store the states that have been visited
    set_visited = set()

    # We will explore the nodes until the queue is empty
    while not in_queue.is_empty():

        # Get the current node from the queue
        current_node = in_queue.pop()

        # Case 1: Check if the current node is the goal state and return the path if it is
        if problem.is_goal_state(current_node.state):
            return current_node.get_path()
        
        # Case 2: Check if the current node has been visited before. If not, add it to the explored set
        if current_node.state not in set_visited:
            set_visited.add(current_node.state)

            # Add the successors of the current node to the queue if they have not been visited before
            for successor in problem.get_successors(current_node.state):

                # In this case, the priority of the nodes is the cost to reach them. successor[2] is the cost to reach the successor
                in_queue.push(SearchNode(current_node, successor), current_node.cost + successor[2])

    # If the goal state is not found, return an empty list
    return []
    # util.raise_not_defined()


    '''
    Question 3 (3 points): Varying the Cost Function
        This UCS implementation  explores nodes by always expanding the one with the lowest total cost first. 
        The solution ensures that Pacman can navigate the maze, considering varying costs for steps such as dangerous ghost-ridden areas
	    or food-rich areas, by adjusting the cost function.

    '''

def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def a_star_search(problem, heuristic=null_heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Initialize the priority queue to store the nodes to be explored
    in_queue = util.PriorityQueue()

    # Initialize the queue with the start state and the path to reach it (empty)
    in_queue.push(SearchNode(None, (problem.get_start_state(), None, 0)), 0)

    # Initialize the explored set that will store the states that have been visited
    set_visited = set()

    # We will explore the nodes until the queue is empty
    while not in_queue.is_empty():

        # Get the current node from the queue
        current_node = in_queue.pop()

        # Case 1: Check if the current node is the goal state and return the path if it is
        if problem.is_goal_state(current_node.state):
            return current_node.get_path()
        
        # Case 2: Check if the current node has been visited before. If not, add it to the explored set
        if current_node.state not in set_visited:
            set_visited.add(current_node.state)

            # Add the successors of the current node to the queue if they have not been visited before
            for successor in problem.get_successors(current_node.state):

                # In this case, the priority of the nodes is the cost to reach them plus the heuristic value. successor[2] is the cost to reach the successor
                # heuristic(successor[0], problem) is the heuristic value of the successor, f(n) = g(n) + h(n).
                in_queue.push(SearchNode(current_node, successor), current_node.cost + successor[2] + heuristic(successor[0], problem))
    
    # If the goal state is not found, return an empty list
    return []
    # util.raise_not_defined()

    '''
    Question 4 (3 points): A* search
        The frontier is initialized as a priority queue, starting with the initial state.
        The algorithm explores nodes until it finds the goal, calculating the combined cost f(n) for each node.
        Successor nodes are added to the frontier with priorities calculated using the cost to reach them (g(n)) and the heuristic function's estimate (h(n)).
        The algorithm returns the optimal path once it reaches the goal.
        A* finds the optimal path to the goal by expanding fewer nodes than UCS, since the heuristic guides the algorithm to the goal more directly.
        UCS tends to expand more nodes because it simply looks at the cumulative cost without having a clear direction of where to go.

    '''

# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search