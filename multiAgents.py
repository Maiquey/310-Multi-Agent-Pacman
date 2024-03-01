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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
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
        newGhostStates = successorGameState.getGhostStates()

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        currentFood = currentGameState.getFood().asList()
        currentPos = currentGameState.getPacmanPosition()

        foodDistances = [util.manhattanDistance(currentPos, food) for food in currentFood]
        closestFoodList = min(foodDistances)
        bestIndices = [index for index in range(len(foodDistances)) if foodDistances[index] == closestFoodList]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        closestFood = currentFood[chosenIndex] # closest food coordinate

        newFoodDistance = util.manhattanDistance(newPos, closestFood)
        score -= newFoodDistance

        for ghostState in newGhostStates:
            newDistance = util.manhattanDistance(newPos, ghostState.getPosition())
            score += newDistance

        return score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        startingAgentIndex = 0
        startingDepth = 0
        self.miniMax(gameState, startingAgentIndex, startingDepth)
        return self.actualAction

    def miniMax(self, state, agentIndex, treeDepth):
        
        if (state.isWin() or state.isLose()):
            return self.evaluationFunction(state)

        #since we are incrementing the agent beyond number of agents we must find the correct agent index
        if (agentIndex >= state.getNumAgents()):
            agentIndex = agentIndex % state.getNumAgents()
        
        if (agentIndex == 0):
            if treeDepth < self.depth:
                #increment treeDepth recursively
                return self.maxValue(state, agentIndex, treeDepth + 1)
            else:
                #terminal value
                return self.evaluationFunction(state)
        else:
            return self.minValue(state, agentIndex, treeDepth)

    def maxValue(self, state, agentIndex, treeDepth):
        v = float('-inf')
        nextAgent = agentIndex + 1
        legalMoves = state.getLegalActions(agentIndex)
        for action in legalMoves:
            successorState = state.generateSuccessor(agentIndex, action)
            successorValue = self.miniMax(successorState, nextAgent, treeDepth)
            v = max(v, successorValue)
            # if top section of minimax, save the action taken with max value
            if treeDepth == 1 and successorValue == v:
                self.actualAction = action
        return v

    def minValue(self, state, agentIndex, treeDepth):
        v = float('inf')
        nextAgent = agentIndex + 1
        legalMoves = state.getLegalActions(agentIndex)
        for action in legalMoves:
            successorState = state.generateSuccessor(agentIndex, action)
            successorValue = self.miniMax(successorState, nextAgent, treeDepth)
            v = min(v, successorValue)
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        startingAgentIndex = 0
        startingDepth = 0
        startingAlpha = float('-inf')
        startingBeta = float('inf')
        self.alphaBetaPrune(gameState, startingAgentIndex, startingDepth, startingAlpha, startingBeta)
        return self.actualAction

    def alphaBetaPrune(self, state, agentIndex, treeDepth, alpha, beta):
        
        if (state.isWin() or state.isLose()):
            return self.evaluationFunction(state)

        #since we are incrementing the agent beyond number of agents we must find the correct agent index
        if (agentIndex >= state.getNumAgents()):
            agentIndex = agentIndex % state.getNumAgents()
        
        if (agentIndex == 0):
            if treeDepth < self.depth:
                #increment treeDepth recursively
                return self.maxValue(state, agentIndex, treeDepth + 1, alpha, beta)
            else:
                #terminal value
                return self.evaluationFunction(state)
        else:
            return self.minValue(state, agentIndex, treeDepth, alpha, beta)

    def maxValue(self, state, agentIndex, treeDepth, alpha, beta):
        v = float('-inf')
        nextAgent = agentIndex + 1
        legalMoves = state.getLegalActions(agentIndex)
        for action in legalMoves:
            successorState = state.generateSuccessor(agentIndex, action)
            successorValue = self.alphaBetaPrune(successorState, nextAgent, treeDepth, alpha, beta)
            v = max(v, successorValue)
            if (v > beta):
                return v
            alpha = max(alpha, v)
            # if top section of minimax, save the action taken with max value
            if treeDepth == 1 and successorValue == v:
                self.actualAction = action
        return v

    def minValue(self, state, agentIndex, treeDepth, alpha, beta):
        v = float('inf')
        nextAgent = agentIndex + 1
        legalMoves = state.getLegalActions(agentIndex)
        for action in legalMoves:
            successorState = state.generateSuccessor(agentIndex, action)
            successorValue = self.alphaBetaPrune(successorState, nextAgent, treeDepth, alpha, beta)
            v = min(v, successorValue)
            if (v < alpha):
                return v
            beta = min(beta, v)
        return v

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
        startingAgentIndex = 0
        startingDepth = 0
        self.expectiMax(gameState, startingAgentIndex, startingDepth)
        return self.actualAction

    def expectiMax(self, state, agentIndex, treeDepth):
        
        if (state.isWin() or state.isLose()):
            return self.evaluationFunction(state)

        #since we are incrementing the agent beyond number of agents we must find the correct agent index
        if (agentIndex >= state.getNumAgents()):
            agentIndex = agentIndex % state.getNumAgents()

        if (agentIndex == 0):
            if treeDepth < self.depth:
                #increment treeDepth recursively
                return self.maxValue(state, agentIndex, treeDepth + 1)
            else:
                #terminal value
                return self.evaluationFunction(state)
        else:
            return self.expValue(state, agentIndex, treeDepth)

    def maxValue(self, state, agentIndex, treeDepth):
        v = float('-inf')
        nextAgent = agentIndex + 1
        legalMoves = state.getLegalActions(agentIndex)
        for action in legalMoves:
            successorState = state.generateSuccessor(agentIndex, action)
            successorValue = self.expectiMax(successorState, nextAgent, treeDepth)
            v = max(v, successorValue)
            # if top section of expectiMax, save the action taken with max value
            if treeDepth == 1 and successorValue == v:
                self.actualAction = action
        return v

    def expValue(self, state, agentIndex, treeDepth):
        v = 0
        nextAgent = agentIndex + 1
        legalMoves = state.getLegalActions(agentIndex)
        for action in legalMoves:
            successorState = state.generateSuccessor(agentIndex, action)
            # assume uniform distribution
            probability = float(1/len(legalMoves))
            successorValue = self.expectiMax(successorState, nextAgent, treeDepth)
            v += probability*(successorValue)
        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Firstly, check if the state is a victory or loss condition, return infinity or negative infinity accordingly
    group food and capsules together, calculate a heuristic to the nearest food/capsule
    check distance from each ghost, if ghosts are scared and close run at them (no specific values chosen for checks, just what seemed to work best)
    if ghosts are running out of scared time, calculate the ghost eval
    at the end, return a weighted score that adds value to finding food and subtracts value if headed near a ghost
    """
    "*** YOUR CODE HERE ***"

    currentGhostStates = currentGameState.getGhostStates()
    score = currentGameState.getScore()
    currentFood = currentGameState.getFood().asList()
    currentPos = currentGameState.getPacmanPosition()
    currentCapsules = currentGameState.getCapsules()

    if currentGameState.isWin():
        return float('inf')
    elif currentGameState.isLose():
        return float('-inf')

    for capsule in currentCapsules:
        currentFood.append(capsule)

    if (len(currentFood) > 0):
        foodDistances = [util.manhattanDistance(currentPos, food) for food in currentFood]
        closestFoodList = min(foodDistances)
        bestIndices = [index for index in range(len(foodDistances)) if foodDistances[index] == closestFoodList]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        closestFood = currentFood[chosenIndex] # closest food coordinate

        foodDistance = util.manhattanDistance(currentPos, closestFood)
        foodEval = float(1/foodDistance)

    for ghostState in currentGhostStates:
        newDistance = util.manhattanDistance(currentPos, ghostState.getPosition())
        if (newDistance < 6 and ghostState.scaredTimer > 4):
            return float('inf')
        elif (newDistance != 0):
            ghostEval = float(1/newDistance)

    return score + 10*foodEval - 5*ghostEval

# Abbreviation
better = betterEvaluationFunction