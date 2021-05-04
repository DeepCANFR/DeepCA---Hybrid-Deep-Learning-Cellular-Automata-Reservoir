import numpy as np


def generate_homogenous_possion_spike_trains(module, length_ms, time_step_ms, firing_rate_hz, nr_of_trains):
    nr_of_time_steps = int(length_ms/time_step_ms)
    random_array = module.random.uniform(0,1,(nr_of_trains, nr_of_time_steps))
    firing_rate_ms = firing_rate_hz / 1000

    probability_of_spike_during_time_step = firing_rate_ms * time_step_ms

    spike_trains = random_array < probability_of_spike_during_time_step
    return spike_trains

def jitter_spike_train(spike_train, jitter, percent):
    spike_positions = np.where(spike_rain == 1)


base_spike_trains = generate_homogenous_possion_spike_trains(np, 2000, 1, 100, 1000)


np.save("spike_trains", base_spike_trains)
