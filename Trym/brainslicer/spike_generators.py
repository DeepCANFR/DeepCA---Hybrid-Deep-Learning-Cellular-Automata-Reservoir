import numpy as ncp

'''
Spike generators
'''


class PoissonSpikeGenerator(object):

    def __init__(self, scale, size, refractory_period):
        self.refractory_period = refractory_period
        self.scale = scale
        self.size = size
        self.last_spike_time = ncp.zeros(size, dtype=ncp.float64)
        self.next_spike_time = self.refractory_period + \
            ncp.random.exponential(self.scale, self.size)

    def homogenous_poisson_spike(self, t):
        new_spikes = self.next_spike_time < t
        new_spikes_mask = new_spikes == 0
        self.last_spike_time *= new_spikes_mask*1.0
        self.last_spike_time += new_spikes*t
        new_next_spike_time = self.refractory_period + \
            ncp.random.exponential(self.scale, self.size)
        self.next_spike_time += new_spikes * new_next_spike_time
        #print(t, self.next_spike_time, new_spikes)
        return new_spikes


if __name__ == "__main__":
    param = {"scale": 1,
             "size": (100, 100),
             "refractory_period": 3}
    psg = PoissonSpikeGenerator(**param)
    print()
