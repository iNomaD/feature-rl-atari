# Handle arguments (before slow imports so --help can be fast)
import argparse
parser = argparse.ArgumentParser(
    description="Train a net to play Atari.")
parser.add_argument("-r", "--render", action="store_true", default=False,
    help="render the game during training or testing")
parser.add_argument("-g", "--game", default="Pong-v0",
    help="game id in gym")
parser.add_argument("-f", "--frames-to-images", action="store_true", default=False,
    help="save frames as .bmp images")
args = parser.parse_args()

import environment
import numpy as np

skip_start = 20

# Make environment
env = environment.GymEnvironment(args.game, args.frames_to_images)
n_outputs = env.numActions()

done = True
episode = 0

while True:
    if done:  # game over, start again
        env.restart()
        for skip in range(skip_start):  # skip the start of each game
            reward, done = env.act(0)
        state = env.getFeatures()
        reward_sum = 0

    if args.render:
        env.render()

    action = np.random.randint(n_outputs)
    reward, done = env.act(action)
    next_state = env.getFeatures()

    state = next_state

    # Compute further statistics for tracking progress (not shown in the book)
    reward_sum += reward
    if done:
        episode += 1
        print('resetting env. episode%d reward %f.' % (episode, reward_sum))