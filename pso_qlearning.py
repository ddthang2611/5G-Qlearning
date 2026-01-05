# pso_qlearning.py
import numpy as np


class PSO_Qlearning:
    def __init__(self, num_particles, dim, fitness_func):
        self.num_particles = num_particles
        self.dim = dim
        self.fitness_func = fitness_func

        self.w = 0.9            
        self.c1 = 1.5
        self.c2 = 1.5

        self.x = np.random.rand(num_particles, dim)
        self.v = np.zeros((num_particles, dim))

        self.pbest = self.x.copy()
        self.pbest_val = np.array([fitness_func(x) for x in self.x])

        self.gbest = self.pbest[np.argmax(self.pbest_val)]
        self.gbest_val = np.max(self.pbest_val)

        self.q_table = np.zeros((3, 3))

        self.alpha = 0.1     # learning rate
        self.gamma = 0.9     # discount factor
        self.epsilon = 0.1   # epsilon-greedy

    def get_state(self, delta_fitness):
        if delta_fitness > 0.7:
            return 2    # cải thiện nhanh
        elif delta_fitness > 0.1:
            return 1    # cải thiện chậm
        else:
            return 0    # kẹt / suy giảm

    def step(self):
        for i in range(self.num_particles):
            old_fit = self.pbest_val[i]

            state = self.get_state(0)

            if np.random.rand() < self.epsilon:
                action = np.random.randint(3)
            else:
                action = np.argmax(self.q_table[state])

            if action == 0:
                self.w = max(0.4, self.w - 0.05)   
            elif action == 2:
                self.w = min(0.9, self.w + 0.1)   

            r1, r2 = np.random.rand(), np.random.rand()
            self.v[i] = (
                self.w * self.v[i]
                + self.c1 * r1 * (self.pbest[i] - self.x[i])
                + self.c2 * r2 * (self.gbest - self.x[i])
            )

            self.x[i] += self.v[i]
            self.x[i] = np.clip(self.x[i], 0.01, 1)

            new_fit = self.fitness_func(self.x[i])

            reward = (new_fit - old_fit) + 0.1 * (self.gbest_val - old_fit)

            delta_fit = new_fit - old_fit
            new_state = self.get_state(delta_fit)

            self.q_table[state, action] += self.alpha * (
                reward
                + self.gamma * np.max(self.q_table[new_state])
                - self.q_table[state, action]
            )

            if new_fit > self.pbest_val[i]:
                self.pbest[i] = self.x[i]
                self.pbest_val[i] = new_fit

        idx = np.argmax(self.pbest_val)
        if self.pbest_val[idx] > self.gbest_val:
            self.gbest = self.pbest[idx]
            self.gbest_val = self.pbest_val[idx]

        return self.gbest_val
