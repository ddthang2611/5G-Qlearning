# pso_qlearning.py
import numpy as np


class PSO_Qlearning:
    def __init__(self, num_particles, dim, fitness_func):
        self.num_particles = num_particles
        self.dim = dim
        self.fitness_func = fitness_func

        # ================= PSO PARAMETERS =================
        # Inertia weight sẽ được Q-learning điều khiển
        self.w = 0.9                 # bắt đầu cao để exploration
        self.c1 = 1.5
        self.c2 = 1.5

        # ================= PARTICLES ======================
        self.x = np.random.rand(num_particles, dim)
        self.v = np.zeros((num_particles, dim))

        self.pbest = self.x.copy()
        self.pbest_val = np.array([fitness_func(x) for x in self.x])

        self.gbest = self.pbest[np.argmax(self.pbest_val)]
        self.gbest_val = np.max(self.pbest_val)

        # ================= Q-LEARNING =====================
        # State: xu hướng cải thiện fitness (3 trạng thái)
        # Action: giảm / giữ / tăng inertia weight
        self.q_table = np.zeros((3, 3))

        self.alpha = 0.1     # learning rate
        self.gamma = 0.9     # discount factor
        self.epsilon = 0.1   # epsilon-greedy

    # =====================================================
    # STATE DEFINITION (ĐÃ SỬA)
    # State phản ánh XU HƯỚNG, không phải fitness tuyệt đối
    # =====================================================
    def get_state(self, delta_fitness):
        if delta_fitness > 0.5:
            return 2    # cải thiện nhanh
        elif delta_fitness > 0:
            return 1    # cải thiện chậm
        else:
            return 0    # kẹt / suy giảm

    # =====================================================
    # ONE ITERATION OF PSO + Q-LEARNING
    # =====================================================
    def step(self):
        for i in range(self.num_particles):
            old_fit = self.pbest_val[i]

            # ---------- STATE ----------
            # Ở đầu iteration, chưa có delta → giả sử trung tính
            state = self.get_state(0)

            # ---------- ACTION SELECTION (ε-greedy) ----------
            if np.random.rand() < self.epsilon:
                action = np.random.randint(3)
            else:
                action = np.argmax(self.q_table[state])

            # ---------- ACTION EFFECT (ĐÃ SỬA) ----------
            # Q-learning điều khiển inertia weight
            if action == 0:
                self.w = max(0.4, self.w - 0.05)   # giảm w → exploitation
            elif action == 2:
                self.w = min(0.9, self.w + 0.1)   # tăng w → exploration
            # action == 1: giữ nguyên w

            # ---------- PSO UPDATE ----------
            r1, r2 = np.random.rand(), np.random.rand()
            self.v[i] = (
                self.w * self.v[i]
                + self.c1 * r1 * (self.pbest[i] - self.x[i])
                + self.c2 * r2 * (self.gbest - self.x[i])
            )

            self.x[i] += self.v[i]
            self.x[i] = np.clip(self.x[i], 0.01, 1)

            # ---------- FITNESS ----------
            new_fit = self.fitness_func(self.x[i])

            # ---------- REWARD (ĐÃ SỬA – RẤT QUAN TRỌNG) ----------
            # Reward = cải thiện cá nhân + đóng góp global
            reward = (new_fit - old_fit) + 0.1 * (self.gbest_val - old_fit)

            # ---------- NEW STATE ----------
            delta_fit = new_fit - old_fit
            new_state = self.get_state(delta_fit)

            # ---------- Q-TABLE UPDATE (Bellman) ----------
            self.q_table[state, action] += self.alpha * (
                reward
                + self.gamma * np.max(self.q_table[new_state])
                - self.q_table[state, action]
            )

            # ---------- PERSONAL BEST ----------
            if new_fit > self.pbest_val[i]:
                self.pbest[i] = self.x[i]
                self.pbest_val[i] = new_fit

        # ---------- GLOBAL BEST ----------
        idx = np.argmax(self.pbest_val)
        if self.pbest_val[idx] > self.gbest_val:
            self.gbest = self.pbest[idx]
            self.gbest_val = self.pbest_val[idx]

        return self.gbest_val
