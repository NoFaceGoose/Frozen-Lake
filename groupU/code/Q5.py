from env.env_frozenlake import FrozenLake
from algorithms.Tabular_model_based.tabular_model_based import policy_evaluation, policy_iteration, value_iteration
from algorithms.Tabular_model_free.q_learning import q_learning
from algorithms.Tabular_model_free.sarsa import sarsa
import contextlib
import numpy as np

@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)
################ Main function ################

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

    print("# Model-free algorithms")
    # change for big lake
    max_episodes = 20000
    eta = 0.99
    epsilon = 0.99

    print("")

    print("## Sarsa")
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    delta1 = (value - policy_evaluation(env, policy, gamma, 0.001, max_episodes))
    print("## Comparison")
    with _printoptions(precision=3, suppress=True):
        print(delta1[:-1].reshape(env.lake.shape))
    print("## Delta mean:", np.absolute(delta1).mean())

    print("")

    print("## Q-learning")
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    delta2 = (value - policy_evaluation(env, policy, gamma, 0.001, max_episodes))
    print("## Comparison")
    with _printoptions(precision=3, suppress=True):
        print(delta2[:-1].reshape(env.lake.shape))

    print(f"episodes:{max_episodes}")
    print(f"learning rate:{eta}")
    print(f"exploration rate:{epsilon}")
    print("## Sarsa")
    print("## Delta mean:", np.absolute(delta1).mean())
    print("## q_learning")
    print("## Delta mean:", np.absolute(delta2).mean())
    


if __name__ == "__main__":
    bigLake()
