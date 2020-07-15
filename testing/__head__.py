import gym
import numpy as np
import torch
import torch.optim as optim
import sys

from agent import BaseAgent

VERTICES = np.array([
    np.array([0.  , 0.           ]),
    np.array([1.  , 0.           ]),
    np.array([1/2., np.sqrt(3)/2.]),
])

def ba2xy(
    triple : np.ndarray,
) -> np.ndarray:
    return VERTICES.T.dot(triple.T).T

def xy2ba(
    x : float,
    y : float,
) -> np.ndarray:
    corner_x = VERTICES.T[0]
    corner_y = VERTICES.T[1]

    x_1 = corner_x[0]
    x_2 = corner_x[1]
    x_3 = corner_x[2]
    y_1 = corner_y[0]
    y_2 = corner_y[1]
    y_3 = corner_y[2]

    l1 = ((y_2 - y_3) * (x   - x_3) + (x_3 - x_2) * (y   - y_3)) / \
         ((y_2 - y_3) * (x_1 - x_3) + (x_3 - x_2) * (y_1 - y_3))
    l2 = ((y_3 - y_1) * (x   - x_3) + (x_1 - x_3) * (y   - y_3)) / \
         ((y_2 - y_3) * (x_1 - x_3) + (x_3 - x_2) * (y_1 - y_3))
    l3 = 1 - l1 - l2

    return np.array([l1, l2, l3])

def save_factors(
    agent : BaseAgent,
):
    X = np.arange(-0.1, 1.1, 0.01)
    Y = np.arange( 0.0, 1.0, 0.01)
    X, Y = np.meshgrid(X, Y)

    triple = xy2ba(X, Y - 0.1)
    transp = np.transpose(triple, axes = [1, 2, 0])

    num_actions = agent.env.mdp.pomdp.num_actions
    dat = np.zeros((*transp.shape[:-1], num_actions))

    for a in range(num_actions):
        print('Computing for action %d of %d ...' % (a + 1, num_actions))

        for i in range(transp.shape[0]):
            # print('Computing for row %d of %d ...' % (i + 1, transp.shape[0]))

            for j in range(transp.shape[1]):
                s = np.concatenate((transp[i, j], np.array([0])))

                dat[i, j, a] = agent.action_value(s, a)

    np.save('volume/decision.plex', dat)
