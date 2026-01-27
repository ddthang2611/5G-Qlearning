import numpy as np

class PSO_Qlearning:
    def __init__(self, num_particles, dim, fitness_func):
        self.num_particles = num_particles
        self.dim = dim
        self.fitness_func = fitness_func
        
        self.c1 = 1.5
        self.c2 = 1.5
        
        self.alpha = 0.5     # Tốc độ học
        self.gamma = 0.9     # Hệ số chiết khấu
        self.epsilon = 0.2   # Tăng nhẹ để hạt thoát khỏi bẫy hội tụ cục bộ 

        self.x = np.random.rand(num_particles, dim)
        self.v = np.zeros((num_particles, dim))

        self.pbest = self.x.copy()
        self.pbest_val = np.array([fitness_func(x) for x in self.x])

        self.gbest = self.pbest[np.argmax(self.pbest_val)].copy()
        self.gbest_val = np.max(self.pbest_val)

        self.q_table = np.zeros((3, 3))

    def get_state(self, delta_fit):
       
        if delta_fit > 0.1: return 2   # Cải thiện tốt
        elif delta_fit > 0: return 1   # Cải thiện nhẹ
        else: return 0                 # Không cải thiện (kẹt)

    def step(self, t, max_iter):
        w_max, w_min = 0.9, 0.4
        w = w_max - ((w_max - w_min) / max_iter) * t

        for i in range(self.num_particles):
            state = self.get_state(self.pbest_val[i] - self.gbest_val)
            
            if np.random.rand() < self.epsilon:
                action = np.random.randint(3)
            else:
                action = np.argmax(self.q_table[state])

            r1, r2 = np.random.rand(), np.random.rand()
            
            q_influence = self.alpha * (self.q_table[state, action] / 100.0) 
            
            self.v[i] = (w * self.v[i] + 
                         self.c1 * r1 * (self.pbest[i] - self.x[i]) + 
                         self.c2 * r2 * (self.gbest - self.x[i]) + 
                         q_influence)

            self.x[i] = np.clip(self.x[i] + self.v[i], 0.01, 1)
            new_fit = self.fitness_func(self.x[i])
            
            reward = new_fit - self.pbest_val[i] 
            print(reward)
            
            new_state = self.get_state(reward)
            self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + \
                                          self.alpha * (reward + self.gamma * np.max(self.q_table[new_state]))

            if new_fit > self.pbest_val[i]:
                self.pbest[i] = self.x[i].copy()
                self.pbest_val[i] = new_fit

        idx = np.argmax(self.pbest_val)
        if self.pbest_val[idx] > self.gbest_val:
            self.gbest = self.pbest[idx].copy()
            self.gbest_val = self.pbest_val[idx]

        return self.gbest_val