import numpy as np
import analysis_module as am
import matplotlib.pyplot as plt

nr_of_distances = 3
nr_of_train_pairs = 10
train_length_in_data_points = 500

distances_along_time = np.zeros((nr_of_distances, nr_of_train_pairs, train_length_in_data_points))
distances = [0.01, 0.02, 0.04]

for index, d in enumerate(distances):

    for i in range(nr_of_train_pairs):
        pattern_0_name = '0' + "_train_nr_" + str(i) + "_network_history_1_heterogenous_Izhikevich_somas_1_distance_by_jitter.npy"
        pattern_1_name = str(d) + "_train_nr_" + str(i) + "_network_history_1_heterogenous_Izhikevich_somas_1_distance_by_jitter.npy"
        pattern_0 = np.load(pattern_0_name)
        pattern_1 = np.load(pattern_1_name)
        # To do: implement import of trains
        print(pattern_1_name)

        convolved_pattern_0 = np.apply_along_axis(am.gaussian_convolution, -1, pattern_0, tau = 5, time_averaged = True)
        convolved_pattern_1 = np.apply_along_axis(am.gaussian_convolution, -1, pattern_1, tau = 5, time_averaged = True)
        #plt.plot(convolved_pattern_0[0,0,0,:])
        #plt.show()
        distance_vector = convolved_pattern_0 - convolved_pattern_1

        flattened_state_history = am.flatten_high_dimensional_state_history(distance_vector, -1)

        distance = np.linalg.norm(flattened_state_history, axis = 1, ord = 2)

        distances_along_time[index,i, :] = distance


#mean_distance_over_time_per_distance = np.mean(distances_along_time)

mean_distance = np.mean(distances_along_time, axis = 1)
print(mean_distance.shape)
for i in range(3):
    plt.plot(mean_distance[i,:], label = str(i) + str(np.mean(mean_distance[i,:])))
plt.legend()
plt.show()
