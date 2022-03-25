# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:07:53 2020

@author: trymlind
"""


import framework_module as fm
import numpy as np
import cv2
#import matplotlib.pyplot as plt


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

neuron_population_excitatory = fm.Soma_AS_with_projection_weights(membrane_decay_excitatory, treshold_decay_excitatory, membrane_treshold_resting_distance_excitatory, population_size_x = population_x, population_size_y = population_y, neighbourhood_template = neighbourhood_template_excitatory, weight_mean = 0.0, weight_SD = 0.3)
#neuron_population_excitatory = fm.Soma_AS_with_projection_weights(membrane_decay = 0.5, treshold_decay = 0.1, membrane_treshold_resting_distance = 0.8, population_size_x = population_x, population_size_y = population_y, neighbourhood_template = neighbourhood_template_excitatory, weight_mean = 0.5/(2*1), weight_SD = 0.3)

#####################################

neighbourhood_template_inhibitory = np.ones((11,11))
neighbourhood_template_inhibitory[2:5,2:5] = 0

neuron_population_inhibitory = fm.Soma_AS_with_projection_weights(membrane_decay = 0.5, treshold_decay = 0.5, membrane_treshold_resting_distance = 1.0, population_size_x = population_x, population_size_y = population_y, neighbourhood_template = neighbourhood_template_inhibitory, weight_mean = 0.2, weight_SD = 0.4)
kill_mask = np.random.rand(population_x, population_y) > 0

last_excitatory = np.zeros((population_x, population_y))

stim_length = 200
input_pattern_on_electrodes = np.random.rand(electrode_x, electrode_y, stim_length)>0.9
#input_pattern = np.zeros((population_x, population_y, stim_length))
#input_pattern[200:205,200:205,:] = 1
t = 0

cap = cv2.VideoCapture(0)
while True:
    if cv2.waitKey(2) & 0xFF == ord('s'):
        t = 0
    if t<stim_length:
    #if np.sum(neuron_population_excitatory.spiked_neurons) == 0:
        input_pattern = np.zeros((population_x, population_y))
        #input_pattern[np.random.randint(0,population_x), np.random.randint(0,population_y)] = 2
        input_pattern[electrode_indexes] = input_pattern_on_electrodes[:,:,t]
        neuron_population_excitatory.membrane_additive_input(input_pattern)#input_pattern[:,:,t])
        t += 1
        print(t)
        
    neuron_population_excitatory.send_spikes_to_synapses(neuron_population_excitatory.spiked_neurons)
    neuron_population_excitatory.membrane_additive_input(np.sum(neuron_population_excitatory.spike_array,2))
    neuron_population_excitatory.compute_spikes()
    
    neuron_population_inhibitory.membrane_additive_input(np.sum(neuron_population_excitatory.spike_array*5,2))
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
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
