'''
Differential equiation solvers
'''
import numpy as ncp

class RungeKutta2(object):
    def __init__(self, f, time_step):
        # Initialize the class with the size of the time steps you are using

        self.f = f
        self.time_step = time_step  # size of time step
        self.t = 0  # starting time

    def advance(self, u, t):

        K1 = self.time_step * (self.f(u, t))
        K2 = self.time_step * self.f(u + (1/2) * K1, t + (1/2)*self.time_step)

        t += self.time_step
        u_delta = K2

        return u_delta


class ForwardEuler(object):
    def __init__(self, f, time_step):
        self.f = f
        self.time_step = time_step  # size of time step
        self.t = 0

    def advance(self):
        k = self.k
        dt = self.time_step
        unew = self.u[k] + dt*self.f(self.u[k], self.t[k])
        return unew
