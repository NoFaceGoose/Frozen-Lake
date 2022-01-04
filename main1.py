from snippets import FrozenLake
# from snippets import policy_evaluation, policy_improvement, policy_iteration, value_iteration
# from snippets import sarsa
# from snippets import linear_q_learning, linear_sarsa, q_learning, LinearWrapper
from snippets import sarsa, q_learning
################ Main function ################


def main():
    seed = 500

    # Small lake
    lake = [
        ["&", ".", ".", "."],
        [".", "#", ".", "#"],
        [".", ".", ".", "#"],
        ["#", ".", ".", "$"],
    ]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)

    print("# Model-based algorithms")
    gamma = 0.9
    theta = 0.001
    max_iterations = 100

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

if __name__ == "__main__":
    main()
