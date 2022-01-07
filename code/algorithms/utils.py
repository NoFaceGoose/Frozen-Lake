import numpy as np

def q_update(eta, q, gamma, state, state_next, reward, action, action_next):
    predict = q[state, action]
    target = reward + gamma * np.max(q[state_next])
    q[state, action] += eta * (target - predict)

def argmax_random(actions, random_state):
    max_value = np.max(actions)
    max_indices = np.flatnonzero(max_value == actions)
    return random_state.choice(max_indices)

def choose_action(env, actions, random_state, epsilon):
    if random_state.uniform(0,1) < epsilon:
        action = random_state.randint(0, env.n_actions)
    else:
        action = argmax_random(actions, random_state)
    return action