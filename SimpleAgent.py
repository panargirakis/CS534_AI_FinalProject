"""
A Deterministic Agent that tries to put the paddle below the ball
"""
import gym
import argparse
import numpy as np

def paddleCol(obs):
    # Returns the center column of the paddle
    # The first few pixel columns are just the left side of the screen, so skip them
    paddleCol = 10;
    while paddleCol < 150:
        pixel = obs[190][paddleCol][0]
        if pixel != 0:
            return paddleCol+7
        paddleCol+=1
    return 94

def ballCol(obs):
    # Returns the column that the ball is in
    # The lowest bricks to start are on row 92, so check in the spaces from row 93 to 187 for the ball
    # if the ball cannot be found in those rows, just recenter the paddle to try to catch it later
    ballRow = 93;
    while ballRow < 188:
        ballCol = 10
        while ballCol < 150:
            pixel = obs[ballRow][ballCol][0]
            if pixel != 0:
                return ballCol
            ballCol+=1
        ballRow+=1
    return -1

def determineAction(paddleCol, ballCol):
    if ballCol == -1:
        return 1
    difference = paddleCol - ballCol
    if difference < 0:
        return 2
    if difference > 0:
        return 3
    return 1


def main():
    np.set_printoptions(threshold=np.inf)
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
    # Start the game by pressing fire
    obs, stepRew, done, info = env.step(1)
    while True:
        # The screen is a 210x160 array of pixels with three values for Atari's RGB
        # The paddle is on rows 188-191 inclusive (initial center is 189.5)
        # The paddle starts on columns 86-101 inclusive, (initial center is 93.5)
        # The lowest bricks to start are on row 91, so check in the spaces from row 92 to 187 for the ball
        # if the ball cannot be found in those rows, just recenter the paddle to try to catch it later
        # and move the paddle based on its location relative to the ball by checking row 190 for where it is
        paddleCenter = paddleCol(obs)
        ballCenter = ballCol(obs)
        #print(paddleCenter, ballCenter)
        obs, stepRew, done, info = env.step(determineAction(paddleCenter, ballCenter))
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
