from evolution.game_models.game import Game, GameThread
import random
import copy
from statistics import mean
import numpy as np
import os
import shutil
from evolution.game_models.base_game_model import BaseGameModel
from evolution.convolutional_neural_network import ConvolutionalNeuralNetwork
import time
# TODO get this running as is on the cartpool enviornment
# TODO things to figure out:
# 1. figure out what the hell it's doing with the is not instance()
# 2. Find out how this stops training

# OTHER THINGS
# Look into changing this into a tensorflow training process
# I'd like to update the code to use linear algebra and use tensorflow to
# quickly run a bunch of matrix operations in parallel instead of what its doing now in sequence
class GEGameModel(BaseGameModel):

    model = None

    def __init__(self, game_name, mode_name, input_shape, action_space, logger_path, model_path):
        BaseGameModel.__init__(self,
                               game_name,
                               mode_name,
                               logger_path,
                               input_shape,
                               action_space)
        self.model_path = model_path
        self.input_shape = input_shape
        self.action_space = action_space
        self.model = ConvolutionalNeuralNetwork(input_shape, action_space).model

    def _predict(self, state):
        q_values = self.model.predict(np.expand_dims(np.array(state, dtype=np.float32), axis=0), batch_size=1)
        return np.argmax(q_values[0])




class GESolver(GEGameModel):

    def __init__(self, game_name, input_shape, action_space):
        testing_model_path = "./output/neural_nets/" + game_name + "/ge/testing/model.h5"
        assert os.path.exists(os.path.dirname(testing_model_path)), "No testing model in: " + str(testing_model_path)
        GEGameModel.__init__(self,
                             game_name,
                             "GE testing",
                             input_shape,
                             action_space,
                             "./output/logs/" + game_name + "/ge/testing/" + self._get_date() + "/",
                             testing_model_path)
        self.model.load_weights(self.model_path)

    def move(self, state):
        return self._predict(state)


class GETrainer(GEGameModel):

    run = 0
    generation = 0
    selection_rate = 0.1
    mutation_rate = 0.01
    population_size = 100
    random_weight_range = 1.0
    parents = int(population_size * selection_rate)

    #TODO remove this later
    avg_score = 0

    def __init__(self, game_name, input_shape, action_space):
        GEGameModel.__init__(self,
                             game_name,
                             "GE training",
                             input_shape,
                             action_space,
                             "./output/logs/" + game_name + "/ge/training/"+ self._get_date() + "/",
                             "./output/neural_nets/" + game_name + "/ge/" + self._get_date() + "/model.h5")
        if os.path.exists(os.path.dirname(self.model_path)):
            shutil.rmtree(os.path.dirname(self.model_path), ignore_errors=True)
        os.makedirs(os.path.dirname(self.model_path))

    def move(self, state):
        pass

    def genetic_evolution(self, env):
        print ("population_size: " + str(self.population_size) +\
              ", mutation_rate: " + str(self.mutation_rate) +\
              ", selection_rate: " + str(self.selection_rate) +\
              ", random_weight_range: " + str(self.random_weight_range))
        population = self._initial_population()
        # create game machines so that it can play in parallel
        self.game_machines = self._create_game_machines(env)

        # TODO change this to something else probably
        training_start_time = time.time()
        while self.avg_score < 200:
            print('{{"metric": "generation", "value": {}}}'.format(self.generation))

            # 1. Selection
            gameplay_time = time.time()
            parents = self._strongest_parents(population, env)
            gameplay_end_time = time.time() - gameplay_time
            print('{{"metric": "gameplay time", "value": {}}}'.format(gameplay_end_time))

            # Saving main model based on the current best two chromosomes
            self._save_model(parents)

            # 2. Crossover (Roulette selection)
            pairs_time = time.time()
            pairs = self._create_pairs(parents)
            pairs_time_end = time.time() - pairs_time
            print('{{"metric": "pairs time", "value": {}}}'.format(pairs_time_end))

            cross_over_time = time.time()
            for i in range(0, self.population_size):
                # do a cross over on the chromosomes/weights
                population[i] = self._crossover(pairs[i][0][0], pairs[i][1][0])
            cross_over_time_end = time.time() - cross_over_time
            print('{{"metric": "crossover time", "value": {}}}'.format(cross_over_time_end))
            # 3. Mutation
            mutation_time = time.time()
            self._mutation(population)

            mutation_time_end = time.time() - mutation_time
            print('{{"metric": "mutation time", "value": {}}}'.format(mutation_time_end))

            self.generation += 1

        training_end_time = time.time() - training_start_time
        print('{{"metric": "training time", "value": {}}}'.format(training_end_time))

    def _create_pairs(self, parents):
        pairs = []
        while len(pairs) != self.population_size:
            pairs.append(self._pair(parents))
        return pairs

    def _pair(self, parents):
        # get the total sum of the parents score
        total_parents_score = sum([x[1] for x in parents])
        # pick a random score between 0 and the sum total of the selected AI's
        pick = random.uniform(0, total_parents_score)
        # TODO look into this could be a problem it looks like it's picking the same parent for the
        # pair at random. 
        pair = [self._roulette_selection(parents, pick), self._roulette_selection(parents, pick)]
        return pair

    #TODO remove this method once replace with new method
    def _roulette_selection(self, parents, pick):
        current = 0
        for parent in parents:
            current += parent[1]
            # go until the current score is > random pick
            if current > pick:
                return parent
        return random.choice(parents) # Fallback

    def _combinations(self, parents):
        combinations = []
        for i in range(0, len(parents)):
            for j in range(i, len(parents)):
                combinations.append((parents[i], parents[j]))
        return combinations

    # returns a list of games equal to the number of CPU's 
    def _create_game_machines(self, env):
        num_cpus = os.cpu_count()
        machines = np.empty((num_cpus,), dtype=object)
        for i in range(0, num_cpus):
            game = Game(
                env=env,
                input_shape=self.input_shape,
                action_space=self.action_space
            )
            machines[i] = game
        return machines

    def _strongest_parents(self, population, env):
        scores_for_chromosomes = []

        for i in range(0, len(population), len(self.game_machines)):
            threads = []
            upper_bound = min(i + len(self.game_machines), len(population))

            for j in range(i, upper_bound):
                # get the current game machine
                game: Game = self.game_machines[j-i]
                # set the weights of the games model
                game.set_weights(population[j])
                # create the thread
                thread = GameThread(game)
                threads.append(thread)
                thread.start()

            # wait for all the threads to complete
            for thread in threads:
                thread.join()

            # add the scores for all the chromosomes
            for k in range(i, upper_bound):
                game: Game = self.game_machines[k-i]
                score = game.score
                scores_for_chromosomes.append(
                    (
                        population[k],
                        score
                    )
                )
        # TODO could replace this with a data strucutre that only holds the top (selection rate) chromosomes and scores
        # This might be equally efficent but it would reduce on memory space
        # the current code sorts it so that the best chromosomes are last
        scores_for_chromosomes.sort(key=lambda x: x[1])
        # cut off everything but the last x% leaving the best at the end of the list
        top_performers = scores_for_chromosomes[-self.parents:]
        top_scores = [x[1] for x in top_performers]
        print('{{"metric": "population", "value": {}}}'.format(mean([x[1] for x in scores_for_chromosomes])))
        print('{{"metric": "top_min", "value": {}}}'.format(min(top_scores)))
        print('{{"metric": "top_avg", "value": {}}}'.format(mean(top_scores)))
        print('{{"metric": "top_max", "value": {}}}'.format(max(top_scores)))
        #TODO remove this
        self.avg_score = mean(top_scores)
        return top_performers # returns the (selection rate parents)

    # adds mutations to layer based on the mutation rate layer
    def _mutate_layer(self, layer: np.ndarray):
        # used to determine which weights get a mutation
        mutation_matrix = np.random.choice(
            a=[0, 1],
            size=layer.shape,
            p=[self.mutation_rate, 1 - self.mutation_rate]
        )
        # used to determine which cells to fill with random numbers
        mutation_matrix_complement = self._create_matrix_complement(
            mutation_matrix
        )
        # creates the random numbers to fill the cells
        random_matrix = np.random.uniform(
            low=-self.random_weight_range,
            high=self.random_weight_range,
            size=layer.shape
        )
        # mutations contains the random numbers
        mutations = mutation_matrix_complement * random_matrix
        # multiply the layer by the mutation matrix to remove weights
        layer = layer * mutation_matrix
        # add the mutaitons to the layer
        layer = layer + mutations
        return layer

    def _mutation(self, base_offsprings: np.ndarray):
        for offspring in base_offsprings:
            for i in range(len(offspring)):
                offspring[i] = self._mutate_layer(offspring[i])

    # takes a mtrix of 0's and 1's and returns the complement where
    # the orignal 1's are now 0's and the 0's are now 1's
    def _create_matrix_complement(self, matrix: np.ndarray):
        negative_1_matrix = np.ones(matrix.shape)
        negative_1_matrix = negative_1_matrix * -1
        # change all 0's to -1's and all 1's to zeros
        complement = matrix + negative_1_matrix
        # change -1's to 1's
        complement = complement * -1
        return complement

    # reutrns a new set of weights by doing a crossover of the two layers
    # randomly combines the x and y chromosomes
    def _crossover_layers(self, x: np.ndarray, y: np.ndarray):
        # this will randomly fill a matrix 50/50 with 0 and 1's
        cross_over_matrix = np.random.choice(
            a=[0, 1],
            size=x.shape,
            p=[0.5, 0.5]
        )
        cross_over_complement = self._create_matrix_complement(
            cross_over_matrix
        )
        # x half of the offspring
        x_half = x * cross_over_matrix
        y_half = y * cross_over_complement

        offspring = x_half + y_half
        return offspring

    # crossover the x and y parents to make a child
    def _crossover(self, x: np.ndarray, y: np.ndarray):
        child = np.empty(shape=(len(x)), dtype=np.ndarray)
        for i in range(0, len(child)):
            child_layer = self._crossover_layers(x[i], y[i])
            child[i] = child_layer
        return child

    def _gameplay_for_chromosome(self, chromosome, env):
        
        self.run += 1
        #self.logger.add_run(self.run)

        self.model.set_weights(chromosome)
        state = env.reset()
        score = 0
        while True:
            action = self._predict(state)
            state_next, reward, terminal, info = env.step(action)
            score += np.sign(reward)
            state = state_next
            if terminal:
                #self.logger.add_score(score)
                return score
    def _initial_population(self):
        weights = self.model.get_weights()
        population = np.empty((self.population_size,), dtype=object)
        new_weights = np.empty((len(weights)), dtype=object)

        for i in range(0, self.population_size):
            for j in range(0, len(weights)):
                layer = weights[j]
                new_weights[j] = np.random.uniform(
                    low=-self.random_weight_range,
                    high=self.random_weight_range,
                    size=layer.shape
                )
            population[i] = np.copy(new_weights)
        return population

    def _random_weight(self):
        return random.uniform(-self.random_weight_range, self.random_weight_range)

    def _save_model(self, parents):
        # Takes the top two performers and makes a deep copy of the weights
        x = copy.deepcopy(parents[-1][0])
        y = copy.deepcopy(parents[-2][0])
        
        best_offsprings = self._crossover(x, y)
        # sets the weights to one of the offsprings
        self.model.set_weights(best_offsprings)
        # saves the model
        self.model.save_weights(self.model_path)
