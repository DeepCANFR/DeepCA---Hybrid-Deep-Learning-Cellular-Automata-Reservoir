# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 21:56:26 2020

@author: trymlind
"""



import numpy as np
import cv2
import framework_module as fm

pop_x = 400
pop_y = 400

neighboyrhood_template_1 = np.ones((9,9))


population_1 = fm.Soma_AS_with_projection_weights(membrane_decay = 0.1, treshold_decay = 0.3, membrane_treshold_resting_distance = 0.5, population_size_x = pop_x, population_size_y = pop_y, neighbourhood_template = neighboyrhood_template_1, weight_mean= 1/900, weight_SD = 0.2)
population_2 = fm.Soma_AS_with_projection_weights(membrane_decay = 0.1, treshold_decay = 0.3, membrane_treshold_resting_distance = 0.5, population_size_x = pop_x, population_size_y = pop_y, neighbourhood_template = neighboyrhood_template_1, weight_mean= 1/900, weight_SD = 0.2)
population_3 = fm.Soma_AS_with_projection_weights(membrane_decay = 0.1, treshold_decay = 0.3, membrane_treshold_resting_distance = 0.5, population_size_x = pop_x, population_size_y = pop_y, neighbourhood_template = neighboyrhood_template_1, weight_mean= 1/900, weight_SD = 0.2)
population_4 = fm.Soma_AS_with_projection_weights(membrane_decay = 0.1, treshold_decay = 0.3, membrane_treshold_resting_distance = 0.5, population_size_x = pop_x, population_size_y = pop_y, neighbourhood_template = neighboyrhood_template_1, weight_mean= 1/900, weight_SD = 0.2)

pop_list = [population_1, population_2, population_3, population_4]
for i0 in range(len(pop_list)):
    weights = pop_list[0].weights 
    pop_list[i0].weights = weights

input_pattern_10 = np.zeros((pop_x, pop_y))
input_pattern_11 = np.zeros((pop_x, pop_y))
input_pattern_01 = np.zeros((pop_x, pop_y))
input_pattern_00 = np.zeros((pop_x, pop_y))

inputpoint_1 = (200,200)
inputpoint_2 = (201,200)
inputpoint_3 = (202,202)

input_patterns = [input_pattern_10, input_pattern_01, input_pattern_00, input_pattern_11]
#input_pattern_00[200,200] = 1

t = 0
k = 0
while True:
    
    if t == 20:
        input_pattern_10[inputpoint_1] = 1

        input_pattern_11[inputpoint_1] = 1
        input_pattern_11[inputpoint_2] = 1

        input_pattern_01[inputpoint_2] = 1
        #input_pattern_00[inputpoint_3] = 1

    
    t += 1
    k += 1
        
    for index, pop in enumerate(pop_list):
        pop.membrane_additive_input(input_patterns[index])
        pop.send_spikes_to_synapses(pop.spiked_neurons)
        pop.membrane_additive_input(np.sum(pop.spike_array, 2))
        
        pop.compute_spikes()
        #pop.send_spikes_to_synapses(pop.spiked_neurons)
        #pop.threshold_additive_input(np.sum(pop_list[index-1].spike_array, 2)**2)
    
    video_1 = np.concatenate((population_1.spiked_neurons, population_2.spiked_neurons), 1)
    video_2 = np.concatenate((population_3.spiked_neurons, population_4.spiked_neurons), 1)
    video = np.uint8(np.concatenate((video_1, video_2), 1))*255
    cv2.imshow('frame', video)
    
    #network.network_state[0,:] = 0
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break