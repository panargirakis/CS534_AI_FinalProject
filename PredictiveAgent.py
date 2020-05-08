"""
A Deterministic Agent that tries to put the paddle below the ball,
but also takes into account the current speed of the ball
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
    # The lowest bricks with same color as the ball to start are on row 62, so check in the spaces from row 63 to 187 for the ball
    # if the ball cannot be found in those rows, just recenter the paddle to try to catch it later
    ballRow = 63;
    while ballRow < 188:
        ballCol = 0
        while ballCol < 160:
            pixel = obs[ballRow][ballCol][0]
            if pixel == 200:
                #print(ballRow)
                return ballCol+1
            ballCol+=1
        ballRow+=1
    return -1

def determineAction(paddleCol, ballCol, ballSpeed):
    if ballCol == -1:
        return 1
    difference = paddleCol - ballCol - ballSpeed
    if difference < -7:
        return 2
    if difference > 7:
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
    ballSpeed = 0
    ballCenter = 0
    while True:
        # The screen is a 210x160 array of pixels with three values for Atari's RGB
        # The paddle is on rows 188-191 inclusive (initial center is 189.5)
        # The paddle starts on columns 86-101 inclusive, (initial center is 93.5)
        # and move the paddle based on its location relative to the ball by checking row 190 for where it is
        paddleCenter = paddleCol(obs)
        ballInfo = ballCol(obs)
        if ballInfo == -1:
           ballSpeed = 0
        else:
           ballSpeed = ballInfo - ballCenter
        ballCenter = ballInfo
        #print(paddleCenter, ballCenter)
        obs, stepRew, done, info = env.step(determineAction(paddleCenter, ballCenter, ballSpeed))
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
