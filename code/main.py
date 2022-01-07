from env.env_frozenlake import FrozenLake
from algorithms.Tabular_model_based.tabular_model_based import policy_iteration, value_iteration
from algorithms.Non_tabular_model_free.linear_q_learning import linear_q_learning
from algorithms.Non_tabular_model_free.linear_sarsa import linear_sarsa
from algorithms.Non_tabular_model_free.linear_wrapper import LinearWrapper
from algorithms.Tabular_model_free.q_learning import q_learning
from algorithms.Tabular_model_free.sarsa import sarsa
################ Main function ################


def smallLake():
    seed = 0

    # Small lake
    lake = [
        ["&", ".", ".", "."],
        [".", "#", ".", "#"],
        [".", ".", ".", "#"],
        ["#", ".", ".", "$"],
    ]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)

    print('\n # Small lake \n')

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


def bigLake():
    seed = 0

    # Big lake
    lake = [
                ['&', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '#', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '#', '#', '.', '.', '.', '#', '.'],
                ['.', '#', '.', '.', '#', '.', '#', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '$']
            ]

    env = FrozenLake(lake, slip=0.1, max_steps=64, seed=seed)

    print('\n # Big lake \n')

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
    # change for big lake
    max_episodes = 10000
    eta = 0.99
    epsilon = 0.99

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

if __name__ == "__main__":
    bigLake()
