import sys
from contextlib import closing

import numpy as np
from io import StringIO

import gym
from gym import error, spaces, utils
from gym.envs.toy_text import discrete
from gym.utils import seeding

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
nstep = 0

MAPS = {
    "8x8": [
        "SFPFFFMW",
        "WFPFTFFG",
        "WFFWFPFW",
        "WFFGFFFW",
        "WFTFFFFW",
        "WFWFFFPW",
        "WFFTFGFW",
        "WFMFFWFD"
    ],
}

class FooEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc = None, map_name = "8x8", given_map = None):
        if given_map == None:
            desc = MAPS[map_name]
        else:
            desc = given_map
        self.desc = desc = np.asarray(desc, dtype = 'c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (-100, 6)

        # Number of available actions
        nA = 4
        # Number of available positions
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        
        def inc(row, col, a):
            row1 = row
            col1 = col
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            newletter = desc[row, col]
            if newletter == b'T':
                T_list = np.where(np.array(desc == b'T').astype('float64').ravel() == 1)[0]
                T_list = np.delete(T_list, np.where(T_list == row * ncol + col))
                new_p = np.random.randint(0, len(T_list))
            elif newletter == b'W':
                row = row1
                col = col1
            global nstep
            nstep += 1
            return row, col
        
        def to_s(row, col):
            # Return the state of the agent
            # s1 is the Manhattan distance between GoBot and the door.
            s1 = abs(row - 7) + abs(col - 7)
            # s2 is the number of steps that GoBot has taken.
            s2 = nstep
            # s3 is the new position of GoBot
            s3 = (row, col)
            return (s1, s2, s3)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            done = (bytes(newletter) == b'M') or (bytes(newletter) == b'P') or (bytes(newletter) == b'D')
            if newletter == b'F' or newletter == b'G' or newletter == b'S':
                reward = 8 / (newstate[0] + 1) - 2 * newstate[1] / 500
            elif newletter == b'W':
                reward = 2 / (newstate[0] + 1) - 4 * newstate[1] / 500
            elif newletter == b'P' or newletter == b'M':
                reward = -100
            elif newletter == b'D':
                reward = 10
            elif newletter == b'T':
                reward = 8 / (newstate[0] + 1) - 2 * newstate[1] / 500
            return newstate, reward, done
        
        for row in range(nrow):
            for col in range(ncol):
                s = row * ncol + col
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'D':
                        li.append((1.0, s, 0, True))
                    else:
                        nrd = update_probability_matrix(row, col, a)
                        li.append((
                            1.0, (nrd[0][-1][0] + 1) * (nrd[0][-1][1] + 1) - 1, nrd[1], nrd[2]
                        ))

        super(FooEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode = 'human', close = False):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight = True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                ["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

