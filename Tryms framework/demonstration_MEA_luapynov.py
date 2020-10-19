# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:07:53 2020

@author: trymlind
"""


import framework_module as fm
import numpy as np
import cv2
import matplotlib.pyplot as plt


'''
############################
'''

electrode_x = 8
electrode_y = 8
electrode_nr = electrode_x * electrode_y
electrode_spacing = 30
upper_left_electrode_location = 100

electrode_indexes = [slice(upper_left_electrode_location, upper_left_electrode_location + electrode_x*electrode_spacing, electrode_spacing), slice(upper_left_electrode_location, upper_left_electrode_location + electrode_y*electrode_spacing, electrode_spacing)]


population_x =  400
population_y = 400

electrode_activity = np.zeros((population_y*2, electrode_nr))



neighbourhood_template_excitatory = np.ones((7,7))


membrane_decay_excitatory = np.random.rand(population_x, population_x)
treshold_decay_excitatory = np.random.rand(population_x, population_x)
membrane_treshold_resting_distance_excitatory = np.random.rand(population_x, population_x)

neuron_population_excitatory = fm.Soma_AS_with_projection_weights(membrane_decay_excitatory, treshold_decay_excitatory, membrane_treshold_resting_distance_excitatory, population_size_x = population_x, population_size_y = population_y, neighbourhood_template = neighbourhood_template_excitatory, weight_mean = 0.5/3, weight_SD = 0.3)
#neuron_population_excitatory = fm.Soma_AS_with_projection_weights(membrane_decay = 0.5, treshold_decay = 0.1, membrane_treshold_resting_distance = 0.8, population_size_x = population_x, population_size_y = population_y, neighbourhood_template = neighbourhood_template_excitatory, weight_mean = 0.5/(2*1), weight_SD = 0.3)

#####################################

neighbourhood_template_inhibitory = np.ones((9,9))
#neighbourhood_template_inhibitory[2:5,2:5] = 0

neuron_population_inhibitory = fm.Soma_AS_with_projection_weights(membrane_decay = 0.5, treshold_decay = 0.5, membrane_treshold_resting_distance = 1, population_size_x = population_x, population_size_y = population_y, neighbourhood_template = neighbourhood_template_inhibitory, weight_mean = 0.9, weight_SD = 0.3)
kill_mask = np.random.rand(population_x, population_y) > 0.8

last_excitatory = np.zeros((population_x, population_y))


# define experiment parameters
nr_of_experiments = 7
recording_length = 200

jitter_spike_time = 50
stim_length = 150

input_pattern_on_electrodes = np.random.rand(electrode_x, electrode_y, stim_length)*0.1 #>0.99
#input_pattern_on_electrodes = np.zeros((electrode_x, electrode_y, stim_length))
input_y = 0
input_pattern_on_electrodes[:,input_y, jitter_spike_time] = 1


# define recording variables
first_recording = np.zeros((electrode_nr, recording_length))
true_first_recording = np.zeros((population_x*population_y, recording_length))
true_second_recording = np.zeros((population_x*population_y, recording_length))

recording_distances = np.zeros((recording_length, nr_of_experiments))
recording_overlaps = np.zeros((recording_length, nr_of_experiments))

true_recording_distances = np.zeros((recording_length, nr_of_experiments))
true_recording_overlaps = np.zeros((recording_length, nr_of_experiments))

# 
stimulation_t = 0
recording_t = 0
experiment_t = 0

recording_nr = 0


first_stimulation = True


cap = cv2.VideoCapture(0)

neuron_population_excitatory.membrane_potentials *= 0
neuron_population_excitatory.thresholds[:,:] = neuron_population_excitatory.m_t_resting_distance
neuron_population_excitatory.spiked_neurons *= 0

neuron_population_inhibitory.membrane_potentials *= 0
neuron_population_inhibitory.thresholds *= 0
neuron_population_inhibitory.thresholds += 1
neuron_population_inhibitory.thresholds[:,:] *= neuron_population_excitatory.m_t_resting_distance
neuron_population_inhibitory.spiked_neurons *= 0

while recording_nr < nr_of_experiments:

    if stimulation_t < stim_length:
   
        input_pattern = np.zeros((population_x, population_y))
        input_pattern[electrode_indexes] = input_pattern_on_electrodes[:,:,stimulation_t]
        neuron_population_excitatory.membrane_additive_input(input_pattern)#input_pattern[:,:,t])
        stimulation_t += 1
        
    print(recording_nr, jitter_spike_time, recording_t, stimulation_t)
        
    
    neuron_population_excitatory.send_spikes_to_synapses(neuron_population_excitatory.spiked_neurons)
    neuron_population_excitatory.membrane_additive_input(np.sum(neuron_population_excitatory.spike_array,2))
    neuron_population_excitatory.compute_spikes()
    
    neuron_population_inhibitory.membrane_additive_input(np.sum(neuron_population_excitatory.spike_array,2))
    neuron_population_inhibitory.compute_spikes()
    neuron_population_inhibitory.spiked_neurons *= kill_mask
    neuron_population_inhibitory.send_spikes_to_synapses(neuron_population_inhibitory.spiked_neurons)
    
    neuron_population_excitatory.threshold_additive_input(np.sum(neuron_population_inhibitory.spike_array,2))
    
    
    electrode_activity = np.roll(electrode_activity,1, axis = 0)
    
    ##############################
    neuron_activity = (neuron_population_excitatory.spiked_neurons + neuron_population_inhibitory.spiked_neurons) >= 1
    electrode_activity_membranes = neuron_activity[electrode_indexes]
    
    electrode_activity[0,:] = electrode_activity_membranes.flatten()
   
    video = np.concatenate((neuron_population_excitatory.spiked_neurons, neuron_population_inhibitory.spiked_neurons))
    video = np.concatenate((video, electrode_activity),axis = 1)
    cv2.imshow('frame', video)
    
    #network.network_state[0,:] = 0
    
    ##########
    #experiment
    ##########
    
    


        
        
    if first_stimulation == False:
        print("second ", recording_nr, jitter_spike_time, recording_t, stimulation_t)
        recording_distances[recording_t, recording_nr] = np.linalg.norm((first_recording[:,recording_t] - electrode_activity_membranes.flatten()), ord = 2)
        recording_overlaps[recording_t, recording_nr] = np.sum((first_recording[:,recording_t] + electrode_activity_membranes.flatten()) == 2)
        
        temp = neuron_activity
        temp = temp.flatten()
        temp = temp + true_first_recording[:,recording_t]
        temp = temp == 2
        temp = np.sum(temp)
        
        true_recording_overlaps[recording_t, recording_nr] = temp
        
        #true_recording_overlaps[second_recording_t, recording_nr] = np.sum((true_first_recording[:,second_recording_t] + np.reshape((neuron_population_inhibitory.spiked_neurons + neuron_population_inhibitory.spiked_neurons)>=1), population_x*population_y) == 2)
        
        temp = neuron_activity
        temp = temp.flatten()
        true_second_recording[:,recording_t] = temp
        temp = true_first_recording[:,recording_t] - temp
        temp = np.linalg.norm(temp,ord = 2)
                
        true_recording_distances[recording_t,recording_nr] = temp
        
        recording_t += 1
        
        print("second ", recording_nr, jitter_spike_time, recording_t, stimulation_t)
        if recording_t == recording_length:
            recording_nr += 1
            #first_stimulation = True
            recording_t = 0
            experiment_t = 0
            stimulation_t = 0
            
            neuron_population_excitatory.membrane_potentials *= 0
            neuron_population_excitatory.thresholds[:,:] = neuron_population_excitatory.m_t_resting_distance
            neuron_population_excitatory.spiked_neurons *= 0
            
            neuron_population_inhibitory.membrane_potentials *= 0
            neuron_population_inhibitory.thresholds *= 0
            neuron_population_inhibitory.thresholds += 1
            neuron_population_inhibitory.thresholds[:,:] *= neuron_population_excitatory.m_t_resting_distance
            neuron_population_inhibitory.spiked_neurons *= 0
            
            
            
            #input_pattern_on_electrodes[5,input_y, jitter_spike_time] = 0
            jitter_spike_time += 0
            input_y += 1
            input_pattern_on_electrodes[:,input_y, jitter_spike_time] = np.random.rand(input_pattern_on_electrodes.shape[0])
            
            experiment_t += 1
            
            
    if first_stimulation:
        print("first ", recording_nr, jitter_spike_time, recording_t, stimulation_t)
        first_recording[:,recording_t] = electrode_activity_membranes.flatten()
        true_first_recording[:,recording_t] = neuron_activity.flatten()
        recording_t += 1
        print("first ", recording_nr, jitter_spike_time, recording_t, stimulation_t)
        if recording_t == recording_length:
            first_stimulation = False
            
            stimulation_t = 0
            recording_t = 0
            
            neuron_population_excitatory.membrane_potentials *= 0
            neuron_population_excitatory.thresholds[:,:] = neuron_population_excitatory.m_t_resting_distance
            neuron_population_excitatory.spiked_neurons *= 0
            
            neuron_population_inhibitory.membrane_potentials *= 0
            neuron_population_inhibitory.thresholds *= 0
            neuron_population_inhibitory.thresholds += 1
            neuron_population_inhibitory.thresholds[:,:] *= neuron_population_excitatory.m_t_resting_distance
            neuron_population_inhibitory.spiked_neurons *= 0
            
    
    
        

        
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

plt.subplot(2,2,1)
plt.plot(true_recording_distances)
plt.subplot(2,2,2)
plt.plot(np.sum(true_recording_distances,0))

plt.subplot(2,2,3)
plt.plot(true_recording_overlaps)
plt.subplot(2,2,4)
plt.plot(np.sum(true_recording_overlaps,0))
plt.show()
