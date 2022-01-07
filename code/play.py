from env.env_frozenlake import FrozenLake

def play(env):
    actions = ["w", "a", "s", "d"]

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input("\nMove: ")
        if c not in actions:
            raise Exception("Invalid action")

        state, r, done = env.step(actions.index(c))

        env.render()
        print("Reward: {0}.".format(r))


seed = 0
# Small lake
lake = [
    ["&", ".", ".", "."],
    [".", "#", ".", "#"],
    [".", ".", ".", "#"],
    ["#", ".", ".", "$"],
]
env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)

play(env)