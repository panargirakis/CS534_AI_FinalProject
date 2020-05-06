"""
A Random Agent
"""
import gym
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
    args = parser.parse_args()

    # Get the environment and extract the number of actions.
    env = gym.make(args.env_name)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n
    obs = env.reset()
    best_rew = float('-inf')
    rew = 0
    while True:
        obs, stepRew, done, info = env.step(env.action_space.sample())
        rew += stepRew
        if rew > best_rew:
            print("new best reward {} => {}".format(best_rew, rew))
            best_rew = rew
        env.render()
        if done:
            rew = 0
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
