import gym
import queue
import math
import random


MEM_SIZE = 1


class RememberBot:

    def __init__(self, observation_size, output_size):
        self.obs_size = observation_size
        self.memory = queue.Queue(maxsize=MEM_SIZE)  # stores the last _ actions
        self.weights = [[random_weight() for i in
                         range((MEM_SIZE + observation_size))]
                        for j in range(output_size)]
        self.outputs = [0 for i in range(output_size)]
        self.score = 0
        self.prob = 0

    def copy(self):
        new = RememberBot(self.obs_size, len(self.outputs))
        new.weights = self.weights
        new.score = self.score
        return new

    def set_weights(self, weights):
        self.weights = weights

    def erase_memory(self):
        self.memory = queue.Queue(maxsize=MEM_SIZE)

    def gererate_action(self, obs):
        inputs = []
        for o in obs:
            inputs.append(o)

        new_mem = queue.Queue(maxsize=MEM_SIZE)
        while not self.memory.empty():
            item = self.memory.get()
            new_mem.put(item)
            inputs.append(item)
        self.memory = new_mem

        self.get_outputs(inputs)
        act = self.get_action()

        if self.memory.full():
            self.memory.get()
        self.memory.put(act)

        return act

    def get_outputs(self, inputs):
        length = len(inputs)
        for i in range(len(self.outputs)):
            val = 0
            for j in range(length):
                val += inputs[j] * self.weights[i][j]
            self.outputs[i] = val

    def get_action(self):
        max_prob = 0
        action = 0
        for i in range(len(self.outputs)):
            prob = self.outputs[i]
            if prob >= max_prob:
                max_prob = prob
                action = i
        return action


def train_bots(env, is_render=False, is_print=False):
    POPULATION_SIZE = 1000
    CULL_PERCENTAGE = 0.05
    CULL_SIZE = int(CULL_PERCENTAGE * POPULATION_SIZE)
    MUTATION_RATE = 0.1
    NUM_GENERATIONS = 20

    NUM_TRIALS = 2

    input_size = len(env.observation_space.high)
    output_size = env.action_space.n

    # Generate the initial generation
    generation = []
    for i in range(POPULATION_SIZE):
        generation.append(RememberBot(input_size, output_size))

    for i in range(NUM_GENERATIONS):
        pop_average = 0

        # Evaluate the fitness of each individual
        for bot in generation:
            score = 0
            for trial in range(NUM_TRIALS):
                if trial == 0 and generation.index(bot) % 100 == 0:
                    score += run_sim(env, bot,
                                     is_render=is_render, is_print=is_print)
                else:
                    score += run_sim(env, bot)

            bot.score = score / NUM_TRIALS
            pop_average += bot.score

        # Cull
        generation.sort(reverse=True, key=lambda x: x.score)
        generation = generation[:CULL_SIZE]

        # Crossover
        generate_probabilities(generation)
        generation = crossover(generation, POPULATION_SIZE)

        # Mutate
        generation = mutate(generation, MUTATION_RATE)

        # Add new random agents
        generation = generation[:int((1 - CULL_PERCENTAGE) * POPULATION_SIZE)]
        for j in range(int(CULL_PERCENTAGE * POPULATION_SIZE)):
            generation.append(RememberBot(input_size, output_size))

        # print some stats for the generation
        if NUM_GENERATIONS >= 100:
            if i % 10 == 0:
                print("Gen", i, "Average score:", pop_average / POPULATION_SIZE)
        else:
            print("Gen", i, "Average score:", pop_average / POPULATION_SIZE)

    # Evaluate the last generation
    for bot in generation:
        bot.score = run_sim(env, bot)
    generation.sort(reverse=True, key=lambda x: x.score)
    return generation[0]


def mutate(gen, chance):
    for p in gen[1:]:
        w = p.weights
        new_w = []
        for col in w:
            new_col = []
            for val in col:
                r = random.random()
                if r < chance:
                    new_col.append(random_weight())
                else:
                    new_col.append(val)
            new_w.append(new_col)
        p.set_weights(new_w)
    return gen


def crossover(gen, pop_size):
    new_gen = []
    rows = len(gen[0].weights)
    cols = len(gen[0].weights[0])
    new_gen.append(gen[0])

    for i in range(1, pop_size):
        w1 = choose_random(gen).weights
        w2 = choose_random(gen).weights
        split = random.randint(0, rows*cols)
        new_w = [[w1[j][k] if ((j*cols) + k < split) else w2[j][k]
                 for k in range(cols)] for j in range(rows)]
        new = gen[0].copy()
        new.set_weights(new_w)
        new_gen.append(new)
    return new_gen


def sigmoid(value):
    return 1 / (1 + math.exp(-1 * value))


def random_weight():
    return random.gauss(0, 10)


def choose_random(generation):
    r = random.random()
    sum_p = 0
    for p in generation:
        sum_p += p.prob
        if sum_p > r:
            return p
    return ValueError("probabilities don't add up")


def generate_probabilities(generation):
    min_s = generation[0].score
    sum_s = 0
    size = len(generation)
    for p in generation:
        s = p.score
        sum_s += s
        if s < min_s:
            min_s = s

    for p in generation:
        if min_s * size == sum_s:
            p.prob = 1/size
        else:
            p.prob = (p.score - min_s) / (sum_s - (min_s * size))
        if p.prob < 0:
            raise ValueError("Negative Probability")


def run_sim(env, mbot, is_render=False, is_print=False):
    mbot.erase_memory()
    observation = env.reset()
    episode_reward = 0
    for t in range(1000):
        if is_render:
            env.render()
        action_i = mbot.gererate_action(observation)
        action = action_i
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            if is_print:
                print("Episode finished after {} timesteps".format(t + 1))
            break

    return episode_reward


def create_bot(env, weights):
    input_size = len(env.observation_space.high)
    output_size = env.action_space.n
    bot = RememberBot(input_size, output_size)
    bot.set_weights(weights)
    return bot


def truncate_weights(weights):
    out = "["
    for i in weights:
        out += "["
        for w in i:
            out += "{0:0.2f}".format(w) + ", "
        out = out[:-2]
        out += "], "
    out = out[:-2]
    out += "]"
    return out


def class_demo():
    num_episodes = 5

    while True:
        # best PoleCart:
        env = gym.make('CartPole-v1')
        weights = [[0.76, -25.65, -10.97, -19.10, 7.88],
                   [-2.45, 3.52, 2.60, 13.23, 10.21]]
        bot = create_bot(env, weights)
        for ep in range(num_episodes):
            run_sim(env, bot, is_render=True, is_print=True)
        env.close()

        # best MointainCar:
        env = gym.make('MountainCar-v0')
        weights = [[-0.48, -29.32, -5.05],
                   [2.19, -2.61, -12.23],
                   [-0.29, 18.69, 6.44]]
        bot = create_bot(env, weights)
        for ep in range(num_episodes):
            run_sim(env, bot, is_render=True, is_print=True)
        env.close()

        # best Acrobot:
        env = gym.make('Acrobot-v1')
        weights = [[7.55, 3.92, -0.61, 10.64, 14.38, -6.75, 1.43],
                   [-13.52, -6.86, -3.85, 2.73, -11.69, -3.35, -24.24],
                   [-0.38, -10.18, 1.03, -7.21, -0.85, 12.99, 2.55]]
        bot = create_bot(env, weights)
        for ep in range(num_episodes):
            run_sim(env, bot, is_render=True, is_print=True)
        env.close()


def run_training():
    # environment = 'CartPole-v1'
    # environment = 'Acrobot-v1'
    environment = 'MountainCar-v0'
    env = gym.make(environment)
    trained_file = open("training_cache", 'a')

    # To use a bot with pre-trained weights comment out this line and
    # uncomment the next two. Also place your weights as a 2d array in the
    # weights variable below.

    best_bot = train_bots(env, is_render=True, is_print=True)

    # best MountainCar:
    # weights = [[-0.48, -29.32, -5.05],
    #            [2.19, -2.61, -12.23],
    #            [-0.29, 18.69, 6.44]]
    # best_bot = create_bot(env, weights)

    sum_reward = 0
    num_episodes = 10

    for episode in range(num_episodes):
        reward = run_sim(env, best_bot, is_render=True, is_print=True)
        sum_reward += reward
    env.close()

    sum_reward = sum_reward / num_episodes
    print("bot performance:", str(sum_reward))

    trained_file.write(environment + ": " + str(sum_reward) + "\n")
    trained_file.write(truncate_weights(best_bot.weights) + "\n\n")
    trained_file.close()


if __name__ == "__main__":
    class_demo()
    # run_training()
