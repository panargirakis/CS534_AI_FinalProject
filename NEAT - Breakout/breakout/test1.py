import gym

env = gym.make('Breakout-v0')

print(env.action_space)

frame = env.reset()
env.render()

is_done = False
while not is_done:
    frame, row, is_done, _ = env.step(env.action_space.sample())
    env.render()