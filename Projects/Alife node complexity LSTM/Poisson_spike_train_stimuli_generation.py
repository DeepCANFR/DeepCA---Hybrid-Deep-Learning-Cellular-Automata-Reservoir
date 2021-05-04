import numpy as np
from numba import njit
import analysis_module as am
import cv2


def generate_homogenous_possion_spike_trains(module, length_ms, time_step_ms, firing_rate_hz, nr_of_trains):
    nr_of_time_steps = int(length_ms/time_step_ms)
    random_array = module.random.uniform(0,1,(nr_of_trains, nr_of_time_steps))
    firing_rate_ms = firing_rate_hz / 1000

    probability_of_spike_during_time_step = firing_rate_ms * time_step_ms

    spike_trains = random_array < probability_of_spike_during_time_step
    spike_trains = spike_trains * 1
    return spike_trains

@njit
def jitter_spike(spike_train, max_jitter):
    spike_positions = np.where(spike_train == 1)
    pos_nr_to_change = np.random.randint(0, len(spike_positions[0]))

    pos_to_change = spike_positions[0][pos_nr_to_change]
    position_change = len(spike_train) + 10
    while pos_to_change == position_change or position_change < 0 or position_change >= len(spike_train):
        spike_positions = np.where(spike_train == 1)
        pos_nr_to_change = np.random.randint(0, len(spike_positions[0]))
        pos_to_change = spike_positions[0][pos_nr_to_change]
        position_change = pos_to_change + np.random.randint(-max_jitter, max_jitter+1)
        if position_change >= 0 and position_change < len(spike_train):
            if spike_train[position_change] == 1:
                position_change = len(spike_train) + 10



    spike_train[pos_to_change] = 0
    spike_train[position_change] = 1

    return spike_train



def jitter_spike_train(spike_train, distance, tolerance, max_jitter):
    convolved_spike_train = am.gaussian_convolution(spike_train)
    jittered_spike_train = np.zeros(len(spike_train))
    jittered_spike_train[:] = spike_train[:]
    train_distance = 0
    while train_distance < distance - tolerance or  train_distance > distance + tolerance:
        jittered_spike_train = jitter_spike(jittered_spike_train, max_jitter)
        convolved_jittered_spike_train = am.gaussian_convolution(jittered_spike_train)
        train_distance = am.d(convolved_spike_train, convolved_jittered_spike_train, 1)
        print(train_distance)
    return jittered_spike_train




if __name__ == "__main__":
    nr_of_spike_trains = 10
    firing_rate_hz = 100
    distances = [0.01, 0.02, 0.04]
    length_ms = 500
    max_jitter = 1

    base_spike_trains = generate_homogenous_possion_spike_trains(np, length_ms, 1, firing_rate_hz, nr_of_spike_trains)

    jittered_spike_trains = np.zeros((nr_of_spike_trains, length_ms, len(distances)))

    for distance_nr in range(len(distances)):
        for train_nr in range(nr_of_spike_trains):
            jittered_spike_trains[train_nr,:,distance_nr] = jitter_spike_train(base_spike_trains[train_nr,:], distances[distance_nr], 0.001, max_jitter)




    np.save("spike_trains_base", base_spike_trains)
    np.save("spike_trains_jittered", jittered_spike_trains)
