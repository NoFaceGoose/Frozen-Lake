from algorithms.Tabular_model_free.q_learning import q_learning
from algorithms.Tabular_model_free.sarsa import sarsa
from env.env_frozenlake import FrozenLake
from algorithms.Tabular_model_based.tabular_model_based import policy_evaluation

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

    print("# Model-free algorithms")
    max_episodes = 189
    eta = 0.5
    epsilon = 0.5

    print("")

    print("## Sarsa")
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    sarsa_value = policy_evaluation(env, policy, gamma, theta, max_iterations)
    env.render(policy, value)

    print("")

    print("## Q-learning")
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    q_value = policy_evaluation(env, policy, gamma, theta, max_iterations)
    env.render(policy, value)

    print("")

    print(f"sarsa_value:\n{sarsa_value}")
    print(f"q_value:\n{q_value}")

if __name__ == "__main__":
    smallLake()
