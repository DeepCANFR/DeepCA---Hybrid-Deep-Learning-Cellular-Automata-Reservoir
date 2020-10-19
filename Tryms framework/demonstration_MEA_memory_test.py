# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:07:53 2020

@author: trymlind
"""


import framework_module as fm
import numpy as np
import cv2
import matplotlib.pyplot as plt


electrode_x = 8
electrode_y = 8
electrode_nr = electrode_x * electrode_y
electrode_spacing = 30
upper_left_electrode_location = 100

electrode_indexes = [slice(upper_left_electrode_location, upper_left_electrode_location + electrode_x*electrode_spacing, electrode_spacing), slice(upper_left_electrode_location, upper_left_electrode_location + electrode_y*electrode_spacing, electrode_spacing)]



population_x = 400
population_y = 400

electrode_activity = np.zeros((population_y*2, electrode_nr))


neighbourhood_template_excitatory = np.ones((7,7))


membrane_decay_excitatory = np.random.rand(population_x, population_x)
treshold_decay_excitatory = np.random.rand(population_x, population_x)
membrane_treshold_resting_distance_excitatory = np.random.rand(population_x, population_x)

neuron_population_excitatory = fm.Soma_AS_with_projection_weights(membrane_decay_excitatory, treshold_decay_excitatory, membrane_treshold_resting_distance_excitatory, population_size_x = population_x, population_size_y = population_y, neighbourhood_template = neighbourhood_template_excitatory, weight_mean = 0.5, weight_SD = 0.3)
#neuron_population_excitatory = fm.Soma_AS_with_projection_weights(membrane_decay = 0.5, treshold_decay = 0.1, membrane_treshold_resting_distance = 0.8, population_size_x = population_x, population_size_y = population_y, neighbourhood_template = neighbourhood_template_excitatory, weight_mean = 0.5/(2*1), weight_SD = 0.3)

#####################################

neighbourhood_template_inhibitory = np.ones((9,9))
#neighbourhood_template_inhibitory[2:5,2:5] = 0

neuron_population_inhibitory = fm.Soma_AS_with_projection_weights(membrane_decay = 0.5, treshold_decay = 0.5, membrane_treshold_resting_distance = 1.0, population_size_x = population_x, population_size_y = population_y, neighbourhood_template = neighbourhood_template_inhibitory, weight_mean = 1, weight_SD = 0.3)
kill_mask = np.random.rand(population_x, population_y) > 0.8

last_excitatory = np.zeros((population_x, population_y))



nr_of_experiments = 40
delay_increase = 5
recording_length = 300

delay = recording_length + 0

first_recording = np.zeros((electrode_nr, recording_length))
true_first_recording = np.zeros((population_x*population_y, recording_length))

recording_distances = np.zeros((recording_length, nr_of_experiments))
recording_overlaps = np.zeros((recording_length, nr_of_experiments))

true_recording_distances = np.zeros((recording_length, nr_of_experiments))
true_recording_overlaps = np.zeros((recording_length, nr_of_experiments))

stim_length = 300
input_pattern_on_electrodes = np.random.rand(electrode_x, electrode_y, stim_length)>0.99
#input_pattern = np.zeros((population_x, population_y, stim_length))
#input_pattern[200:205,200:205,:] = 1
stimulation_t = 0
recording_nr = 0
first_recording_t = 0
second_recording_t = 0
experiment_t = 0

first_stimulation = True

cap = cv2.VideoCapture(0)
while recording_nr < nr_of_experiments:

    if stimulation_t<stim_length:
    #if np.sum(neuron_population_excitatory.spiked_neurons) == 0:
        input_pattern = np.zeros((population_x, population_y))
        #input_pattern[np.random.randint(0,population_x), np.random.randint(0,population_y)] = 2
        input_pattern[electrode_indexes] = input_pattern_on_electrodes[:,:,stimulation_t]
        neuron_population_excitatory.membrane_additive_input(input_pattern)#input_pattern[:,:,t])
        
    print(recording_nr, experiment_t, stimulation_t, first_recording_t, second_recording_t)
        
    neuron_population_excitatory.send_spikes_to_synapses(neuron_population_excitatory.spiked_neurons)
    neuron_population_excitatory.membrane_additive_input(np.sum(neuron_population_excitatory.spike_array,2))
    neuron_population_excitatory.compute_spikes()
    
    neuron_population_inhibitory.membrane_additive_input(np.sum(neuron_population_excitatory.spike_array,2))
    neuron_population_inhibitory.compute_spikes()
    neuron_population_inhibitory.spiked_neurons *= kill_mask
    neuron_population_inhibitory.send_spikes_to_synapses(neuron_population_inhibitory.spiked_neurons)
    
    neuron_population_excitatory.threshold_additive_input(np.sum(neuron_population_inhibitory.spike_array,2))
    
    electrode_activity = np.roll(electrode_activity,1, axis = 0)
    #electrode_activity_membranes = neuron_population_excitatory.membrane_potentials[electrode_indexes] + neuron_population_inhibitory.membrane_potentials[electrode_indexes]
    electrode_activity_membranes = neuron_population_excitatory.spiked_neurons[electrode_indexes] + neuron_population_inhibitory.spiked_neurons[electrode_indexes]
    
    electrode_activity[0,:] = np.reshape(electrode_activity_membranes, electrode_x * electrode_y)
   
    video = np.concatenate((neuron_population_excitatory.spiked_neurons, neuron_population_inhibitory.spiked_neurons))
    video = np.concatenate((video, electrode_activity),axis = 1)
    cv2.imshow('frame', video)
    
    #network.network_state[0,:] = 0
    
    ##########
    #experiment
    ##########
    
    if first_stimulation:
        first_recording[:,first_recording_t] = np.reshape(electrode_activity_membranes, electrode_x * electrode_y)
        true_first_recording[:,first_recording_t] = np.reshape((neuron_population_excitatory.spiked_neurons + neuron_population_inhibitory.spiked_neurons) >= 1, population_x*population_y)
        first_recording_t += 1
        stimulation_t += 1
        if first_recording_t == recording_length:
            first_stimulation = False
            first_recording_t = 0
            
            
    if experiment_t == delay:
        stimulation_t = 0
        
    if experiment_t >= delay:
        recording_distances[second_recording_t,recording_nr] = np.linalg.norm(first_recording[:,second_recording_t] - np.reshape(electrode_activity_membranes, electrode_x * electrode_y), ord = 2)
        recording_overlaps[second_recording_t, recording_nr] = np.sum((first_recording[:,second_recording_t] + np.reshape(electrode_activity_membranes, electrode_x * electrode_y)) == 2)
        
        temp = neuron_population_inhibitory.spiked_neurons + neuron_population_inhibitory.spiked_neurons
        temp = temp >= 1
        temp = temp.flatten()
        print(temp)
        temp = temp + true_first_recording[:,second_recording_t]
        temp = temp == 2
        temp = np.sum(temp)
        
        true_recording_overlaps[second_recording_t, recording_nr] = temp
        
        #true_recording_overlaps[second_recording_t, recording_nr] = np.sum((true_first_recording[:,second_recording_t] + np.reshape((neuron_population_inhibitory.spiked_neurons + neuron_population_inhibitory.spiked_neurons)>=1), population_x*population_y) == 2)
        
        temp = neuron_population_inhibitory.spiked_neurons + neuron_population_inhibitory.spiked_neurons
        temp = temp >= 1
        temp = temp.flatten()
        temp = true_first_recording[:,second_recording_t] - temp
        temp = np.linalg.norm(temp,ord = 2)
                
        true_recording_distances[second_recording_t,recording_nr] = temp
        
        
        
        second_recording_t += 1
        stimulation_t += 1
        
        if second_recording_t == recording_length:
            recording_nr += 1
            first_stimulation = True
            second_recording_t = 0
            experiment_t = 0
            delay += delay_increase
            
            stimulation_t = 0
            
            neuron_population_excitatory.membrane_potentials *= 0
            neuron_population_excitatory.thresholds[:,:] = neuron_population_excitatory.m_t_resting_distance
            neuron_population_excitatory.spiked_neurons *= 0
            
            neuron_population_inhibitory.membrane_potentials *= 0
            neuron_population_inhibitory.thresholds[:,:] = neuron_population_excitatory.m_t_resting_distance
            neuron_population_inhibitory.spiked_neurons *= 0
    
    experiment_t += 1
        

        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

plt.figure()
plt.title("Electrodes")
plt.subplot(2,2,1)
plt.plot(recording_distances)
plt.subplot(2,2,2)
plt.plot(np.sum(recording_distances,0))

plt.subplot(2,2,3)
plt.plot(recording_overlaps)
plt.subplot(2,2,4)
plt.plot(np.sum(recording_overlaps,0))

plt.figure()
plt.figure("True")
plt.subplot(2,2,1)
plt.plot(true_recording_distances)
plt.subplot(2,2,2)
plt.plot(np.sum(true_recording_distances,0))

plt.subplot(2,2,3)
plt.plot(true_recording_overlaps)
plt.subplot(2,2,4)
plt.plot(np.sum(true_recording_overlaps,0))
plt.show()
