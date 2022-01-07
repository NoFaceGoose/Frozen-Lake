import numpy as np
from ..utils import choose_action

def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)
        a = choose_action(env, q, random_state, epsilon[i])
        done = False

        while not done:
            features_next, reward, done = env.step(a)
            delta = reward - q[a]
            q = np.dot(features_next, theta)
            action_next = choose_action(env, q, random_state, epsilon[i])
            delta += gamma * q[action_next]
            theta += eta[i] * delta * features[a]
            features = features_next
            a = action_next

    return theta