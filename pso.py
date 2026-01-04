# pso.py
import numpy as np

class PSO:
    def __init__(self, num_particles, dim, fitness_func):
        self.num_particles = num_particles
        self.dim = dim
        self.fitness_func = fitness_func

        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5

        self.x = np.random.rand(num_particles, dim)
        self.v = np.zeros((num_particles, dim))

        self.pbest = self.x.copy()
        self.pbest_val = np.array([fitness_func(x) for x in self.x])

        self.gbest = self.pbest[np.argmax(self.pbest_val)]
        self.gbest_val = np.max(self.pbest_val)

    def step(self):
        for i in range(self.num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            self.v[i] = (
                self.w * self.v[i]
                + self.c1 * r1 * (self.pbest[i] - self.x[i])
                + self.c2 * r2 * (self.gbest - self.x[i])
            )
            self.x[i] += self.v[i]
            self.x[i] = np.clip(self.x[i], 0.01, 1)

            fit = self.fitness_func(self.x[i])
            if fit > self.pbest_val[i]:
                self.pbest[i] = self.x[i]
                self.pbest_val[i] = fit

        idx = np.argmax(self.pbest_val)
        if self.pbest_val[idx] > self.gbest_val:
            self.gbest = self.pbest[idx]
            self.gbest_val = self.pbest_val[idx]

        return self.gbest_val
