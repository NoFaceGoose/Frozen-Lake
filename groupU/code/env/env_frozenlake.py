from env.env_model import Environment

import numpy as np
import contextlib

@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

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

        super(FrozenLake, self).__init__(n_states, n_actions, max_steps, pi, seed)

        # initialize p_table (probability table)
        self.p_table = np.zeros((n_states, n_states, n_actions), dtype=float)

        # assgin each value in p_table
        for state in range(n_states):
            # for the absorbing, hole and goal state, the next state must be absorbing state
            if state == self.absorbing_state or self.lake_flat[state] in ('#', '$'):
                self.p_table[state, self.absorbing_state, :] = 1
                continue
            
            # get the position of current state
            row, col = int(state /  self.cols), state %  self.cols
            
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

    def step(self, action):
        state, reward, done = Environment.step(self, action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    def p(self, next_state, state, action):
        return self.p_table[state][next_state][action]

    def r(self, next_state, state, action=None):
        # print(f"state:{state}")
        # print(f"state_len:{len(self.lake_flat)}")
        if state == self.absorbing_state:
            return 0
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