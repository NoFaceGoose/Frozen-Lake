import numpy as np


################ Model-based algorithms ################

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)
    iteration_times = 0
    stop = False
    while iteration_times < max_iterations and not stop:
        delta = 0
        for state in range(env.n_states):
            action_sum = 0
            current_value = value[state]
            action = policy[state]
            for next_state in range(env.n_states):
                action_sum += env.p(next_state, state, action) * ((value[next_state]*gamma) + env.r(next_state, state, action))
            value[state] = action_sum
            delta = max(delta, abs(current_value - value[state]))

        iteration_times += 1
        if delta < theta:
            stop = True
    return value

def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)

    for state in range(env.n_states):
        action_sum = np.zeros((env.n_actions))
        for next_state in range(env.n_states):
            for action in range(env.n_actions):
                action_sum[action] += ((value[next_state] * gamma) + env.r(next_state, state))*env.p(next_state, state, action)
        policy[state] = np.argmax(action_sum)

    return policy


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    value = np.zeros(env.n_states, dtype=np.float)

    iteration_times = 0
    
    while iteration_times < max_iterations:
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy = policy_improvement(env, value, gamma)
        iteration_times += 1

    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    policy = np.zeros(env.n_states, dtype=int)
    
    iteration_times = 0
    stop = False
    while iteration_times < max_iterations and not stop:
        delta = 0
        for state in range(env.n_states):
            action_sum = np.zeros((4))
            current_value = value[state]
            for next_state in range(env.n_states):
                for action in range(env.n_actions):
                    action_sum[action] += ((value[next_state] * gamma) + env.r(next_state, state))*env.p(next_state, state, action)
            value[state] = np.max(action_sum)
            delta = max(delta, abs(current_value-value[state]))
        iteration_times += 1
        if delta < theta:
            stop = True
    return policy, value