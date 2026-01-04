# network_model.py
import numpy as np

class NetworkModel:
    def __init__(self, num_embb, num_urllc, num_rb):
        self.num_embb = num_embb
        self.num_urllc = num_urllc
        self.num_users = num_embb + num_urllc
        self.num_rb = num_rb
        self.channel_gain = np.random.uniform(0.5, 2.0, self.num_users)

    def evaluate(self, allocation):
        rb_alloc = allocation / np.sum(allocation) * self.num_rb
        rb_alloc = np.maximum(rb_alloc, 1e-3)

        sinr = self.channel_gain / 0.1
        throughput = rb_alloc * np.log2(1 + sinr)

        latency = np.zeros(self.num_users)
        latency[:self.num_embb] = 4 + 10 / rb_alloc[:self.num_embb]
        latency[self.num_embb:] = 1 + 5 / rb_alloc[self.num_embb:]

        energy = 0.5 * rb_alloc

        urllc_latency = latency[self.num_embb:]
        reliability = np.mean(urllc_latency < 10)

        return {
            "throughput": np.sum(throughput),
            "latency": np.mean(latency),
            "energy": np.sum(energy),
            "reliability": reliability
        }
