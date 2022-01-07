import numpy as np
from ..utils import choose_action, q_update

def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        action = choose_action(env, q[s], random_state, epsilon[i])
        done = False

        while not done:
            state_next, reward, done = env.step(action)
            action_next = choose_action(env, q[state_next], random_state, epsilon[i])
            q_update(eta[i], q, gamma, s, state_next, reward, action, action_next)
            s = state_next
            action = action_next

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value