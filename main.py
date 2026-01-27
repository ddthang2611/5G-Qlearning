# main_pso_qlearning.py
import matplotlib.pyplot as plt
from network_model import NetworkModel
from pso import PSO
from pso_qlearning import PSO_Qlearning

num_embb = int(input("Nhập số lượng eMBB users: "))
num_urllc = int(input("Nhập số lượng URLLC users: "))
num_rb = 50

net = NetworkModel(
    num_embb=num_embb, 
    num_urllc=num_urllc, 
    num_rb=num_rb)

dim = net.num_users
iter_num = 100

def fitness(x):
    r = net.evaluate(x) 
    # Fitness = T - (0.5*L + 0.3*E) theo phương trình (4.1.3) [cite: 335, 418]
    return r["throughput"] - (0.5 * r["latency"] + 0.3 * r["energy"])

pso = PSO(num_particles=10, dim=dim, fitness_func=fitness)
pso_q = PSO_Qlearning(num_particles=10, dim=dim, fitness_func=fitness)

fit_pso, fit_pso_q = [], []

for i in range(iter_num):
    fit_pso.append(pso.step())
    fit_pso_q.append(pso_q.step(i, iter_num)) 

plt.plot(fit_pso, label="Standard PSO")
plt.plot(fit_pso_q, label="Q-learning PSO (Proposed)")
plt.xlabel("Iteration")
plt.ylabel("Fitness Value")
plt.legend()
plt.show()
