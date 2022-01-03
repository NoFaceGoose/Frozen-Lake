from snippets import FrozenLake
# from snippets import policy_evaluation, policy_improvement, policy_iteration, value_iteration
# from snippets import sarsa
# from snippets import linear_q_learning, linear_sarsa, q_learning, LinearWrapper
from snippets import policy_iteration
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

    print("# Model-based algorithms")
    gamma = 0.9
    theta = 0.001
    max_iterations = 100

    print("")

    print("## Policy iteration")
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print("")

if __name__ == "__main__":
    main()
