import numpy as np
import os
import random
import shutil
from game_models.base_game_model import BaseGameModel
from convolutional_neural_network import ConvolutionalNeuralNetwork

GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 100000
BATCH_SIZE = 32
TRAINING_FREQUENCY = 4
TOTAL_STEP_UPDATE_FREQUENCY = 10000
TARGET_NETWORK_UPDATE_FREQUENCY = 40000
MODEL_PERSISTENCE_UPDATE_FREQUENCY = 10000
REPLAY_START_SIZE = 50000

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_TEST = 0.05
EXPLORATION_STEPS = 1000000 #TODO: ???
EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS


class DDQNGameModel(BaseGameModel):

    def __init__(self, game_name, mode_name, observation_space, action_space, path):
        BaseGameModel.__init__(self, game_name,
                               mode_name,
                               path,
                               observation_space,
                               action_space)

        self.input_shape = (4, observation_space, observation_space)

        self.model_path = "./output/neural_nets/" + game_name + "/ddqn/model.h5"

        self.action_space = action_space

        self.ddqn = ConvolutionalNeuralNetwork(self.input_shape, action_space, LEARNING_RATE).model
        self._load_model()
        self.memory = []

    def _save_model(self):
        self.ddqn.save_weights(self.model_path)

    def _load_model(self):
        if os.path.isfile(self.model_path):
            self.ddqn.load_weights(self.model_path)


class DDQNSolver(DDQNGameModel):

    def __init__(self, game_name, observation_space, action_space):
        DDQNGameModel.__init__(self, game_name, "DDQN testing", observation_space, action_space, "./output/logs/" + game_name + "/ddqn/testing/")

    def move(self, state):
        if np.random.rand() < EXPLORATION_TEST:
            return random.randrange(self.action_space)
        q_values = self.ddqn.predict(np.asarray([state]).astype(np.float64), batch_size=1)
        return np.argmax(q_values[0])


class DDQNTrainer(DDQNGameModel):

    def __init__(self, game_name, observation_space, action_space):
        DDQNGameModel.__init__(self, game_name, "DDQN training", observation_space, action_space, "./output/logs/" + game_name + "/ddqn/training/")

        if os.path.exists(os.path.dirname(self.model_path)):
            shutil.rmtree(os.path.dirname(self.model_path), ignore_errors=True)
        os.makedirs(os.path.dirname(self.model_path))

        self.ddqn_target = ConvolutionalNeuralNetwork(self.input_shape, action_space, LEARNING_RATE).model
        self._reset_target_network()
        self.epsilon = EXPLORATION_MAX

    def move(self, state):
        if np.random.rand() < self.epsilon or len(self.memory) < REPLAY_START_SIZE:
            return random.randrange(self.action_space)
        q_values = self.ddqn.predict(np.asarray([state]).astype(np.float64), batch_size=1)
        return np.argmax(q_values[0])

    def remember(self, current_state, action, reward, next_state, terminal):
        self.memory.append({"current_state": np.asarray([current_state]),
                            "action": action,
                            "reward": reward,
                            "next_state": np.asarray([next_state]),
                            "terminal": terminal})
        if len(self.memory) > MEMORY_SIZE:
            self.memory.pop(0)

    def step_update(self, total_step):
        if len(self.memory) >= REPLAY_START_SIZE:
            if total_step % TRAINING_FREQUENCY == 0:
                loss, accuracy = self._train()
                self.logger.add_loss(loss)
                self.logger.add_accuracy(accuracy)

            self._update_epsilon()

            if total_step % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0 and total_step >= MODEL_PERSISTENCE_UPDATE_FREQUENCY:
                self._save_model()

            if total_step % TARGET_NETWORK_UPDATE_FREQUENCY == 0 and total_step >= TARGET_NETWORK_UPDATE_FREQUENCY:
                self._reset_target_network()

        if total_step % TOTAL_STEP_UPDATE_FREQUENCY == 0:
            print('{{"metric": "epsilon", "value": {}}}'.format(self.epsilon))
            print('{{"metric": "total_step", "value": {}}}'.format(total_step))

    def _train(self):
        batch = np.asarray(random.sample(self.memory, BATCH_SIZE))
        if len(batch) < BATCH_SIZE:
            return

        current_states = []
        q_values = []

        for entry in batch:
            current_states.append(entry["current_state"].astype(np.float64))
            next_state = entry["next_state"].astype(np.float64)
            next_state_prediction = self.ddqn_target.predict(next_state).ravel()
            next_q_value = np.max(next_state_prediction)
            q = list(self.ddqn.predict(entry["current_state"])[0])
            if entry["terminal"]:
                q[entry["action"]] = entry["reward"]
            else:
                q[entry["action"]] = entry["reward"] + GAMMA * next_q_value
            q_values.append(q)

        fit = self.ddqn.fit(np.asarray(current_states).squeeze(),
                            np.asarray(q_values).squeeze(),
                            batch_size=BATCH_SIZE,
                            verbose=0)
        loss = fit.history["loss"][0]
        accuracy = fit.history["acc"][0]
        return loss, accuracy

    def _update_epsilon(self):
        self.epsilon -= EXPLORATION_DECAY
        self.epsilon = max(EXPLORATION_MIN, self.epsilon)

    def _reset_target_network(self):
        self.ddqn_target.set_weights(self.ddqn.get_weights())



