"""
Used to evaluate an AI in a gym enviornment
"""
from evolution.convolutional_neural_network import ConvolutionalNeuralNetwork
import threading
from gym import Env
import copy
import numpy as np

class Game():

    def __init__(self, env: Env, input_shape, action_space, weights=None):
        self.env = copy.deepcopy(env)
        # TODO fix this
        self.model = ConvolutionalNeuralNetwork(input_shape, action_space).model
        if(weights is not None):
            self.model.set_weights(weights)
        self.score = 0

    def set_weights(self, weights):
        self.model.set_weights(weights)
        # reset the score
        self.score = 0

    def _predict(self, state):
        q_values = self.model.predict(
            np.expand_dims(
                np.array(
                    state,
                    dtype=np.float32
                ), axis=0), batch_size=1)
        return np.argmax(q_values[0])


class GameThread(threading.Thread):
    def __init__(self, game: Game):
        threading.Thread.__init__(self)
        self.game = game

    def run(self):
        state = self.game.env.reset()
        terminal = False
        while not terminal:
            action = self.game._predict(state)
            state_next, reward, terminal, info = self.game.env.step(action)
            self.game.score += np.sign(reward)
            state = state_next
