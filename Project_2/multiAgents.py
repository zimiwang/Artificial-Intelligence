# multiAgents.py
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

import math
from util import manhattanDistance
from game import Directions
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

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Get ghost distance between pacman
        ghostDistances = []
        for ghost in newGhostStates:
            if ghost.scaredTimer == 0:
                ghostDistances.append(manhattanDistance(newPos, ghost.configuration.pos))

        minmumGhostDist = min(ghostDistances, default=100)
        n_food = successorGameState.getNumFood()
        food = currentGameState.getFood()
        if minmumGhostDist == 0:
            return -math.inf
        if n_food == 0:
            return math.inf

        if food[newPos[0]][newPos[1]]:
            minFoodDist = 0
        else:
            foodDistances = []
            for x in range(food.width):
                for y in range(food.height):
                    if food[x][y]:
                        foodDistances.append(manhattanDistance(newPos, (x, y)))
            minFoodDist = min(foodDistances)

        return -(1 / (minmumGhostDist - 0.8)) + 1 / (minFoodDist + 0.5)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        ghostIndex = range(1, gameState.getNumAgents())

        def min_value(state, d, ghost):
            if state.isWin() or state.isLose() or d == self.depth:
                return self.evaluationFunction(state)

            v = math.inf
            for successor in state.getLegalActions(ghost):
                if ghost == ghostIndex[-1]:
                    v = min(v, max_value(state.generateSuccessor(ghost, successor), d + 1))
                else:
                    v = min(v, min_value(state.generateSuccessor(ghost, successor), d, ghost + 1))
            return v

        def max_value(state, d):
            if state.isWin() or state.isLose() or d == self.depth:
                return self.evaluationFunction(state)

            v = -math.inf
            for successor in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, successor), d, 1))
            return v

        scores = {}
        successors = []
        for successor in gameState.getLegalActions(0):
            next_state = gameState.generateSuccessor(0, successor)
            score = min_value(next_state, 0, 1)
            scores[successor] = score
            successors.append(successor)

        successors.sort(key=lambda s: scores[s], reverse=True)

        return successors[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        ghostIndex = range(1, gameState.getNumAgents())

        def min_value(state, d, ghost, alpha, beta):
            if state.isWin() or state.isLose() or d == self.depth:
                return self.evaluationFunction(state)

            v = math.inf
            for successor in state.getLegalActions(ghost):
                if ghost == ghostIndex[-1]:
                    v = min(v, max_value(state.generateSuccessor(ghost, successor), d + 1, alpha, beta))
                else:
                    v = min(v, min_value(state.generateSuccessor(ghost, successor), d, ghost + 1, alpha, beta))

                if v < alpha:
                    return v
                beta = min(beta, v)

            return v

        def max_value(state, d, alpha, beta):
            if state.isWin() or state.isLose() or d == self.depth:
                return self.evaluationFunction(state)

            v = -math.inf
            for successor in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, successor), d, 1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)

            return v

        final_action = None
        v = -math.inf
        alpha = -math.inf
        beta = math.inf

        for successor in gameState.getLegalActions(0):
            act = min_value(gameState.generateSuccessor(0, successor), 0, 1, alpha, beta)
            if v < act:
                v = act
                final_action = successor
            alpha = max(alpha, act)
            if v >= beta:
                break

        return final_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        ghostIndex = range(1, gameState.getNumAgents())

        def avg_value(state, d, ghost):

            if state.isWin() or state.isLose() or d == self.depth:
                return self.evaluationFunction(state)

            v = 0
            probability = 1 / len(state.getLegalActions(ghost))

            for successor in state.getLegalActions(ghost):
                if ghost == ghostIndex[-1]:
                    v += probability * max_value(state.generateSuccessor(ghost, successor), d + 1)
                else:
                    v += probability * avg_value(state.generateSuccessor(ghost, successor), d, ghost + 1)
            return v

        def max_value(state, d):

            if state.isWin() or state.isLose() or d == self.depth:
                return self.evaluationFunction(state)

            v = -math.inf
            for successor in state.getLegalActions(0):
                v = max(v, avg_value(state.generateSuccessor(0, successor), d, 1))
            return v

        scores = {}
        successors = []
        for successor in gameState.getLegalActions(0):
            next_state = gameState.generateSuccessor(0, successor)
            score = avg_value(next_state, 0, 1)
            scores[successor] = score
            successors.append(successor)

        successors.sort(key=lambda s: scores[s], reverse=True)

        return successors[0]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: based on number 1, added situation values about closestghostdistance, capsules, etc.
    """
    "*** YOUR CODE HERE ***"
    # Calculate the Manhattan distance from the current location to the food
    def food_distance(x):
        return manhattanDistance(x, newPos)

    # Calculate the Manhattan distance from the current location to the ghost
    def ghost_distance(y):
        return manhattanDistance(y, newPos)

    # Initialize score and variables
    score = 0
    ghostPos = []
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    closestFood = sorted(newFood, key=lambda x: manhattanDistance(x, newPos))

    # Get the location of the ghost and add it to the list
    for ghostState in newGhostStates:
        x, y = ghostState.getPosition()
        ghostPos.append((x, y))

    # Sort the locations of the ghosts to get the closest ghost
    closestGhost = sorted(ghostPos, key=lambda y: manhattanDistance(y, newPos))

    # If the current state is a failed state or the current position coincides with the ghost, return negative infinity
    if currentGameState.isLose() or newPos in ghostPos:
        return -math.inf

    # If the distance between the nearest ghost and the current location is less than 3, deduct 300 points
    if ghost_distance(closestGhost[0]) < 3:
        score -= 300

    # If the nearest ghost is less than 2 away from the current location, deduct 1000 points
    if ghost_distance(closestGhost[0]) < 2:
        score -= 1000

    # Returns negative infinity if the distance of the nearest ghost from the current location is less than 1
    if ghost_distance(closestGhost[0]) < 1:
        return -math.inf

    if len(currentGameState.getCapsules()) < 2:
        score += 100

    # If the nearest food or ghost list is empty, add 10 points to the score of the evaluation function
    if len(closestFood) == 0 or len(closestGhost) == 0:
        score += scoreEvaluationFunction(currentGameState) + 10

    # Otherwise, compute a score for the current state, consisting of the sum of the evaluation function score,
    # the distance to the nearest food, and the reciprocal of the distances to the nearest and farthest ghosts
    else:
        score += (scoreEvaluationFunction(currentGameState) +
                  10 / food_distance(closestFood[0]) +
                  1 / ghost_distance(closestGhost[0]) +
                  1 / ghost_distance(closestGhost[-1]))

    return score


# Abbreviation
better = betterEvaluationFunction
