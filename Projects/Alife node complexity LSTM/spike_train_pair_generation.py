import numpy as np
import analysis_module as am

train_length_ms = 500
sampling_rate = 1 #per ms
sampling_points = int(train_length_ms/sampling_rate)
print(sampling_points)
spiking_frequency = 40 #hz
spiking_frequency_per_ms = spiking_frequency/1000
spiking_probability_on_sample_point = spiking_frequency_per_ms/sampling_rate

nr_of_train_pairs = 200
distance_ratios = [0.01, 0.02, 0.04]

distance_tolerance = 0.001

trains =[]
average_spike_frequency = []
for distance_ratio_nr in range(len(distance_ratios)):
  train_list = []
  sum_spiking_frequency = 0
  while len(train_list) < nr_of_train_pairs:
    spiking_frequency = np.random.uniform(10,100) #hz
    spiking_frequency_per_ms = spiking_frequency/1000
    spiking_probability_on_sample_point = spiking_frequency_per_ms/sampling_rate

    train_1 = np.random.uniform(0,1,sampling_points) < spiking_probability_on_sample_point
    train_2 = np.random.uniform(0,1,sampling_points) < spiking_probability_on_sample_point
    #print(np.sum(train_1), np.sum(train_2))
    convolved_train_1 = am.gaussian_convolution(train_1)
    convolved_train_2 = am.gaussian_convolution(train_2)
    #plt.plot(convolved_train_1)
    #plt.plot(convolved_train_2)
    #plt.show()
    distance = am.d(convolved_train_1,convolved_train_2, sampling_frequency = 1, time_averaged = True)
    #print(distance)

    if np.abs(distance - distance_ratios[distance_ratio_nr]) < distance_tolerance:
      sum_spiking_frequency += spiking_frequency
      train_list.append([train_1, train_2])
      print(distance, distance_ratios[distance_ratio_nr])
      print(distance)
  average_spike_frequency.append(sum_spiking_frequency/len(train_list))

  trains.append(train_list)

average_spike_frequency = np.array(average_spike_frequency)
np.save("average_spike_frequency", average_spike_frequency)

for distance_nr, distance in enumerate(distance_ratios):
    for pair_nr in range(len(trains[distance_nr])):
        filename = "distance_ " + str(distance) +"pair_nr_" +str(pair_nr)+ "_" + "train_0"
        np.save(filename, trains[distance_nr][pair_nr][0])
        filename = "distance_ " + str(distance) +"pair_nr_" +str(pair_nr)+ "_" + "train_1"
        np.save(filename, trains[distance_nr][pair_nr][1])
