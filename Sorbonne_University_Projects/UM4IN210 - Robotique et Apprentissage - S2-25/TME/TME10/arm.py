#################################################################################################
#                                                                                               #
#                                                                                               #
#                Evolutionary playground: from convergent to divergent search                   #
#                                                                                               #
#                                                                                               #
#################################################################################################
#                                                                                               #
#                                                                                               #
#   Copyright (C) 2020 Stephane Doncieux, Sorbonne Université                                   #
#                                                                                               #
#  This program is free software; you can redistribute it and/or modify it under the terms      #
#  of the GNU General Public License as published by the Free Software Foundation;              #
#  either version 2 of the License, or (at your option) any later version.                      #
#                                                                                               #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;    #
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    #
#  See the GNU General Public License for more details.                                         #
#                                                                                               #
#  You should have received a copy of the GNU General Public License along with this program;   #
#  if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,             #
#  Boston, MA 02111-1307 USA                                                                    #
#                                                                                               #
#                                                                                               #
#################################################################################################
#                                                                                               #
# This code allows to run different variants of gradient free direct policy search algorithms   #
# It relies on the DEAP framework to allow an easy exploration of EA components (selection,     #
# mutation, ...), see https://deap.readthedocs.io for more details.                             #
#                                                                                               #
# To use it, set the env_name variable below and launch it with python:                                                        #
#       python3 ea_dps.py                                                                        #
#                                                                                               #
# If you have multiple cores on your computer, consider using scoop, it will parallelize        #
# the run and thus greatly accelerate it:                                                       #
#       python3 -m scoop ea_dps.py                                                               #
#                                                                                               #
#################################################################################################

import numpy as np
from numpy import pi, cos, sin
from matplotlib import pyplot as plt
# Computing segment intersection (from https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/)


def ccw(A, B, C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


class Arm:
    def __init__(self, lengths, walls=[], target_pos=[0, 0]):
        self.n_dofs = len(lengths)
        self.lengths = np.concatenate(([0], lengths))
        self.joint_xy = []
        self.walls = np.array(walls)
        self.target_pos = target_pos

    def fw_kinematics(self, p):
        assert (len(p) == self.n_dofs)
        p = np.append(p, 0)
        self.joint_xy = np.zeros((self.n_dofs + 1, 2))
        mat = np.matrix(np.identity(4))
        for i in range(0, self.n_dofs + 1):
            m = [[cos(pi*p[i]), -sin(pi*p[i]), 0, self.lengths[i]],
                 [sin(pi*p[i]),  cos(pi*p[i]), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]
            mat = mat * np.matrix(m)
            v = mat * np.matrix([0, 0, 0, 1]).transpose()
            self.joint_xy[i] = v[0:2].A.flatten()
        inters = False
        for ip in range(len(self.joint_xy)-1):
            for iw in range(self.walls.shape[0]):
                inters = intersect(
                    self.joint_xy[ip], self.joint_xy[ip+1], self.walls[iw, 0], self.walls[iw, 1])
                if inters:
                    break
            if inters:
                break
        return self.joint_xy[self.n_dofs], inters

    def display_configuration(self):
        plt.figure(figsize=(8, 8))
        plt.plot(self.joint_xy[:, 0], self.joint_xy[:, 1], '-o')
        for i in range(self.walls.shape[0]):
            plt.plot(
                self.walls[i, :, 0], self.walls[i, :, 1], color="black", linewidth=2)
        plt.scatter(self.target_pos[0], self.target_pos[1],
                    color="red", marker="x", s=100)
        lim = np.sum(self.lengths)
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.show()
