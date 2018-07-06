"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import time
import sys
import matplotlib.pyplot as plt
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 8  # grid height
MAZE_W = 8  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()
        self.success_times = 0
        self.failed_times = 0
        self.T = []
        self.Success_Rate_list = []
        self.fig  = plt.figure()
        self.ax   = self.fig.add_subplot(2,1,1)

    def  add_hell(self, origin, y, x):
        hell_center = origin + np.array([UNIT*x, UNIT * y])
        hell = self.canvas.create_rectangle(
            hell_center[0] - 15, hell_center[1] - 15,
            hell_center[0] + 15, hell_center[1] + 15,
            fill='black')
        return hell

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        # hell
        self.hell_set = [self.canvas.coords(self.hell1)]
        self.hell_set.append(self.canvas.coords(self.hell2))
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 0, 4)))
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 1, 6)))
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 2, 3)))
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 2, 4))) #
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 2, 5)))
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 2, 6)))
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 3, 1)))
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 3, 3)))
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 4, 1))) #
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 4, 5)))
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 4, 7)))
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 5, 1)))
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 5, 2)))
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 5, 4)))
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 6, 1)))
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 6, 2)))
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 6, 3)))
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 6, 4)))
        self.hell_set.append(self.canvas.coords(self.add_hell(origin, 5, 6)))

        # create oval
        # oval_center = origin + UNIT * 2
        oval_center = origin + np.array([UNIT * 2, UNIT * 2])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        orig2 = origin + np.array([UNIT * 0, UNIT * 0])
        self.rect = self.canvas.create_rectangle(
            orig2[0] - 15, orig2[1] - 15,
            orig2[0] + 15, orig2[1] + 15,
            fill='red')

        # self.rect = self.canvas.create_rectangle(
        #     origin[0] - 15, origin[1] - 15,
        #     origin[0] + 15, origin[1] + 15,
        #     fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        # create red rect
        orig2 = origin + np.array([UNIT * 0, UNIT * 0])
        self.rect = self.canvas.create_rectangle(
            orig2[0] - 15, orig2[1] - 15,
            orig2[0] + 15, orig2[1] + 15,
            fill='red')
        #==========
        # self.rect = self.canvas.create_rectangle(
        #     origin[0] - 15, origin[1] - 15,
        #     origin[0] + 15, origin[1] + 15,
        #     fill='red')

        # return observation
        return [0.0, 0.0]
        # return self.canvas.coords(self.rect)

    def recover_act(self, action):
        if action == 0 or action == 2:
            s_, q_state = self.exe_act(action+1)
        else:
            s_, q_state = self.exe_act(action-1)
        return s_, q_state 

    def exe_act(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:     # forward
            if s[1] > UNIT:
                base_action[1] -= UNIT

        elif action == 1:   # backward
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT

        elif action == 2:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
            
        elif action == 3:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.rect)  # next state 
        q_state = [ (s_[1]-5)/40, (s_[0]-5)/40 ]
        # q_state = s_
        # s_=a
        return s_, q_state
        

    def step(self, action):
        s_, q_state = self.exe_act(action)
        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            self.success_times += 1
            print '\nsuccess / failed = ' + str(self.success_times) + ' / ' + str(self.failed_times)+\
                  ' => success rate = ' + str(1.0*self.success_times/(self.success_times+self.failed_times))
            done = 'success'

        elif s_ in self.hell_set:
            reward = -1
            self.failed_times += 1
            print '\nsuccess / failed = ' + str(self.success_times) + ' / ' + str(self.failed_times)+\
                  ' => success rate = ' + str(1.0*self.success_times/(self.success_times+self.failed_times))
            done = 'failed'
            s_, q_state = self.recover_act(action)

        else:
            reward = 0
            done = 'searching'

        ## Draw "Time-Success_Rate" figure
        # if self.success_times==0:
        #     tmp = 0
        # else:
        #     tmp = 1.0*self.success_times/(self.success_times+self.failed_times)
        #     self.Success_Rate_list.append(tmp)
        #     self.T.append(self.success_times+self.failed_times)
        # if(self.success_times+self.failed_times) == 3:
        #     self.ax.plot(self.T, self.Success_Rate_list, 'r-', lw=3)
        #     plt.pause(0.1)

        return q_state, reward, done
        # return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


