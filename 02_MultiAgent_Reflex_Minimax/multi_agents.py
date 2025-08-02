# multi_agents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattan_distance
from game import Directions, Actions
from pacman import GhostRules
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghostState.scared_timer for ghostState in new_ghost_states]
        
        "*** YOUR CODE HERE ***"    
        # We need to get the list of remaining food, capsules, and ghost positions in the game 
        remaining_food = new_food.as_list()  
        remaining_capsules = successor_game_state.get_capsules()
        ghost_positions = [ghost_state.configuration.pos for ghost_state in new_ghost_states]

        # Now, we compute the minimum distance to the closest food, ghost, and capsule using Manhattan distance
        # To make the code more optimal and clean, we define the function `min_distance` below
        distance_food = min_distance(new_pos, remaining_food)
        distance_ghost = min_distance(new_pos, ghost_positions)
        distance_capsule = min_distance(new_pos, remaining_capsules)

        # After this, we need to check the distances of food, ghosts, and capsules to assign rewards or penalties
        # Get the score from the successor game state
        score = successor_game_state.get_score()
        
        # First, check if there is a closest food; if so, we reward Pacman
        # The number 10 is chosen to give **higher priority** to food, meaning Pacman should prioritize eating food
        # The reward increases as Pacman gets closer to the food (by dividing 10 by the distance)
        # The larger the number, the stronger the incentive to get food
        if distance_food:
            score += 10 / distance_food

        # Second, check the capsules
        if distance_capsule:
            score += 5 / distance_capsule  # Capsules are less important than food, so the reward is lower

        # Finally, check the closest ghost and penalize Pacman for being near a ghost
        if distance_ghost:
                score -= 10 / distance_ghost  

        # Return the final score, which will guide the decision of which action to take
        return score

        #return successor_game_state.get_score()

def min_distance(position, points):
    """
    Calculate the minimum Manhattan distance from the given position to a list of points.
    If there are no points, return 0.
    """
    return min([manhattan_distance(position, point) for point in points]) if points else 0

def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        super().__init__()
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth) 

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluation_function.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generate_successor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.get_num_agents():
        Returns the total number of agents in the game

        game_state.is_win():
        Returns whether or not the game state is a winning state

        game_state.is_lose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # We create the recursive function for the minimax algorithm
        def minimax(game_state, depth, agent_index):

            # Base case: we check for the terminal state (win, lose) or depth limit has been reached
            if depth == 0 or game_state.is_win() or game_state.is_lose():
                return self.evaluation_function(game_state)

            # First case: we check if the current agent is Pacman (maximizing player)
            if agent_index == 0:

                # Call the max_value function for Pacman's move
                return max_value(game_state, depth, agent_index)

            # Second case: we check if the current state agent is Ghost (minimizing player)
            else:
  
                # Call the min_value function for a ghost's move
                return min_value(game_state, depth, agent_index)

        # We have create the max and min values separated for better strucuture and funcionality for the code
        def max_value(game_state, depth, agent_index):
            v = -float('inf')  # First, we initialize v to negative infinity for maximizing

            # After, we iterate over all legal actions Pacman can take
            for action in game_state.get_legal_actions(agent_index):

                # Finally, recursively call minimax for the next agent and update v with the maximum value
                v = max(v, minimax(game_state.generate_successor(agent_index, action), depth, agent_index + 1))

            return v # And return the best score found for Pacman

        def min_value(game_state, depth, agent_index):
            v = float('inf') # First, we initialize v to positive infinity for minimizing
            next_agent = agent_index + 1 # Move to the next agent's turn.

            # We check if all agents have moved, reset to Pacman (agent 0) and decrease depth since we have processed the
            # last action of the ghosts in this round
            if next_agent >= game_state.get_num_agents():
                next_agent = 0
                depth -= 1

            # Now, we iterate over all legal actions the ghost can take
            for action in game_state.get_legal_actions(agent_index):

                # Then, we recursively call minimax for the next agent and update v with the minimum value
                v = min(v, minimax(game_state.generate_successor(agent_index, action), depth, next_agent))

            return v # And we return the worst score for Ghost (best for the Pacman)
        
        # Get the legal actions for Pacman 
        agent_index = 0
        legal_actions = game_state.get_legal_actions(agent_index)
        
        # We evaluate the minimax score for each possible action Pacman can take
        scores = [minimax(game_state.generate_successor(agent_index, action), self.depth, agent_index + 1) for action in legal_actions]

        # We find the best score
        best_score = max(scores)

        # Identify all indices in the scores list where the score matches the best score
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]

        # Randomly select one of the best actions to introduce variability in Pacman's behavior
        chosen_index = random.choice(best_indices)

        # Finally, the fucntion return the action corresponding to the chosen index.
        return legal_actions[chosen_index]

        #util.raise_not_defined()
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function.
        This implementation uses alpha-beta pruning to efficiently explore the game tree.
        """
        
        # We create the recursive function for alpha-beta pruning
        def alpha_beta_pruning(game_state, depth, agent_index, alpha, beta):

            # Base case: Check if we've reached the maximum depth or the game is over (win or lose)
            if depth == 0 or game_state.is_win() or game_state.is_lose():
                return self.evaluation_function(game_state)
            
            # First case: if it is Pacman's turn (Maximizing agent)
            if agent_index == 0:

                # Call the max_value function for Pacman's move
                return max_value(game_state, depth, agent_index, alpha, beta)
            
            # Second case: if it is one of the ghost's turn (Minimizing agent)
            else:
                # Call the min_value function for a ghost's move
                return min_value(game_state, depth, agent_index, alpha, beta)
            
        # The structure is like the minimax function, but we separate the max and min value functions 
        # to optimize the code and use alpha-beta pruning.
        def max_value(game_state, depth, agent_index, alpha, beta):

            v = -float('inf')  # We initialize the best value as negative infinity (for maximization)
            
            # Explore all legal actions for Pacman
            for action in game_state.get_legal_actions(agent_index):

                # Recursively call alphaBeta algorithm for the next agent and update v with the maximum value
                v = max(v, alpha_beta_pruning(game_state.generate_successor(agent_index, action), depth, agent_index + 1, alpha, beta))
                
                # Prune if the value of the current node is greater than or equal to beta (no need to explore further)
                if v > beta:
                    return v
                
                # Update alpha to the maximum value of v and alpha
                alpha = max(alpha, v)
            
            return v
        
        def min_value(game_state, depth, agent_index, alpha, beta):
            v = float('inf')  # We initialize the best value as positive infinity (for minimization)
            next_agent = agent_index + 1
            
            # If all agents have moved, switch back to Pacman (agent 0) and decrease the depth
            if next_agent >= game_state.get_num_agents():
                next_agent = 0
                depth -= 1
            
            # Explore all legal actions for the current ghost
            for action in game_state.get_legal_actions(agent_index):

                v = min(v, alpha_beta_pruning(game_state.generate_successor(agent_index, action), depth, next_agent, alpha, beta))
                
                # Prune if the value of the current node is less than or equal to alpha (no need to explore further)
                if v < alpha:
                    return v
                
                # Update beta to the minimum value of v and beta
                beta = min(beta, v)
            
            return v
        
        # Now, we evaluate the alpha-beta pruning score for each action that Pacman can take
        # We initialize alpha and beta values for alpha-beta pruning
        alpha = -float('inf')  # Start with alpha as negative infinity (worst possible score for maximizer)
        beta = float('inf')  # Start with beta as positive infinity (worst possible score for minimizer)
        
        # Initialize the best_action variable to store the best move Pacman can take
        best_action = None
        
        # Iterate over all legal actions Pacman can take in the current state
        for action in game_state.get_legal_actions(0):

            # For each action, generate the successor state and evaluate it using alpha-beta pruning
            score = alpha_beta_pruning(game_state.generate_successor(0, action), self.depth, 1, alpha, beta)
            
            # If the score of the current action is better than the current alpha value (best score for Pacman)
            if score > alpha:
                alpha = score  # Update alpha to reflect the new best score
                best_action = action  # Store the action corresponding to the new best score
        
        # Finally, return the action that leads to the highest score (the best action for Pacman)
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

def better_evaluation_function(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()
    


# Abbreviation
better = better_evaluation_function
