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
    r = net.evaluate(x, last_call=False) 
    return (
        0.4 * r["throughput"]
        - 0.3 * r["latency"]
        - 0.2 * r["energy"]
        + 0.1 * r["reliability"] * 100
    )

pso = PSO(num_particles=10, dim=dim, fitness_func=fitness)
pso_q = PSO_Qlearning(num_particles=10, dim=dim, fitness_func=fitness)

fit_pso = []
fit_pso_q = []

for i in range(iter_num):
    fit_pso.append(pso.step())
    fit_pso_q.append(pso_q.step())

plt.plot(fit_pso, label="PSO")
plt.plot(fit_pso_q, label="PSO + Q-learning")
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.legend()
plt.show()
