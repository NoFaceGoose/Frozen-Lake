################ Environment ################

import numpy as np
import contextlib
import random

# Configures numpy print options


@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1.0 / n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception("Invalid action.")

        self.n_steps += 1
        done = self.n_steps >= self.max_steps

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        shape = self.lake.shape
        self.rows, self.cols = shape[0], shape[1]

        self.slip = slip

        n_states = self.lake.size + 1
        n_actions = 4

        # translation to the postion of each action(up, left, down, right) 
        self.actions_row =(-1, 0, 1 ,0)
        self.actions_col =(0, -1, 0 ,1)

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == "&")[0]] = 1.0

        self.absorbing_state = n_states - 1

        # initialize p_table (probability table)
        self.p_table = np.zeros((n_states, n_states, n_actions), dtype=float)

        # assgin each value in p_table
        for state in range(n_states):
            # get the position of current state
            row, col = int(state /  self.cols), state %  self.cols

            # for the absorbing, hole and goal state, the next state must be absorbing state
            if state == self.absorbing_state or self.lake_flat[state] in ('#', '$'):
                self.p_table[state, self.absorbing_state, :] = 1
                continue
            
            # assign and accumulate the value for each action
            for action in range(n_actions):
                for slip_action in range(n_actions):
                    # the state will not change by default
                    next_state = state
                    # calculate the new position
                    next_row, next_col = row + self.actions_row[slip_action], col + self.actions_col[slip_action]
                    # the state will change only when the new postion is still in the grid
                    if 0 <= next_row < self.rows and 0 <= next_col < self.cols:
                        # get the state of the new postion
                        next_state = (self.cols * next_row) + next_col

                    # the basic average slip probabiity of current state, next_state and action
                    self.p_table[state, next_state, action] += self.slip / n_actions

                    # assign the probability of no slipping to the current state, next_state and action
                    if action == slip_action:
                        self.p_table[state, next_state, action] += 1 - self.slip
                    
        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed)

    def step(self, action):
        state, reward, done = Environment.step(self, action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    def p(self, next_state, state, action):
        return self.p_table[state][next_state][action]

    def r(self, next_state, state, action):
        if self.lake_flat[state] == "$":
            return 1
        else:
            return 0

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = "@"

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ["^", "<", "_", ">"]

            print("Lake:")
            print(self.lake)

            print("Policy:")
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print("Value:")
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))


def play(env):
    actions = ["w", "a", "s", "d"]

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input("\nMove: ")
        if c not in actions:
            raise Exception("Invalid action")

        state, r, done = env.step(actions.index(c))

        env.render()
        print("Reward: {0}.".format(r))


seed = 0
# Small lake
lake = [
    ["&", ".", ".", "."],
    [".", "#", ".", "#"],
    [".", ".", ".", "#"],
    ["#", ".", ".", "$"],
]
env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
play(env)

################ Model-based algorithms ################

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)

    # TODO:

    return value


def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)

    # TODO:

    return policy


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    # TODO:

    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    # TODO:

    return policy, value


################ Tabular model-free algorithms ################


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        # TODO:

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        # TODO:

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


################ Non-tabular model-free algorithms ################


class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()

        q = features.dot(theta)

        # TODO:

    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()

        # TODO:

    return theta


################ Main function ################


def main():
    seed = 0

    # Small lake
    lake = [
        ["&", ".", ".", "."],
        [".", "#", ".", "#"],
        [".", ".", ".", "#"],
        ["#", ".", ".", "$"],
    ]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)

    print(env.p_table)

    print("# Model-based algorithms")
    gamma = 0.9
    theta = 0.001
    max_iterations = 100

    print("")

    print("## Policy iteration")
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print("")

    print("## Value iteration")
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print("")

    print("# Model-free algorithms")
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5

    print("")

    print("## Sarsa")
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print("")

    print("## Q-learning")
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print("")

    linear_env = LinearWrapper(env)

    print("## Linear Sarsa")

    parameters = linear_sarsa(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print("")

    print("## Linear Q-learning")

    parameters = linear_q_learning(
        linear_env, max_episodes, eta, gamma, epsilon, seed=seed
    )
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
