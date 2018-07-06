"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable
from numpy import loadtxt,savetxt,zeros
import pandas as pd


def update():
    times = 3
    cnt = 0
    while cnt < 5:
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            q_table = RL.learn(str(observation), action, reward, str(observation_), done)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done == 'success':
                break
        pd.to_pickle(q_table, 'q_table_2.pickle')
    # end of game
    print cnt
    print('game over')
    env.destroy()

if __name__ == "__main__":
    
    
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    # RL = QLearningTable(actions=['Forward', 'Backward', 'Left', 'Right'])

    env.after(1000, update)
    env.mainloop()
