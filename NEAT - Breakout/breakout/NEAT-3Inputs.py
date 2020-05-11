import gym
import numpy as np
import cv2
import neat
import pickle
import random
import noise

env = gym.make('BreakoutDeterministic-v4')
np.random.seed(123)
env.seed(123)
imageArr = []
np.set_printoptions(threshold=np.inf)

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
    return -10000

def eval_genomes(genomes, config):
    ballSpeed = 0
    ballCenter = 0
    for genome_id,genome in genomes:
        ob = env.reset()
        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        fitness_current = 0

        frame = 0
        counter = 0
        done = False
        
        env.step(1)
        env.render()
        while not done:
            frame += 1
            factor = 0.5
            #print(ob)
            #ob = np.uint8(noise.noisy(ob, factor))
            #ob = cv2.resize(ob, (inx,iny))
            #ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)

            #imageArr = np.ndarray.flatten(ob)
            imageArr = ob
            paddleCenter = paddleCol(ob)
            ballInfo = ballCol(ob)
            if ballInfo == -1:
                ballSpeed = 0
            else:
                ballSpeed = ballInfo - ballCenter
            ballCenter = ballInfo
            nnOutput = net.activate([paddleCenter, ballCenter, ballSpeed])

            numerical_input = nnOutput.index(max(nnOutput))
            ob, rew, done, info = env.step(numerical_input)

            fitness_current += rew

            if(rew > 0):
                counter = 0
            else:
                counter += 1

            if (counter == 15):
                counter = 0
                ob, rew, done, info  = env.step(1)
                fitness_current += rew
            env.render()

            if(done):
                #done = True
                print(genome_id, fitness_current)
                #ob, rew, done, info = env.step(1)

            genome.fitness = fitness_current


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward.txt')

p = neat.Population(config)

p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open('winner_breakout_neat_1.pkl', 'wb') as output:
    pickle.dump(winner, output,1)
