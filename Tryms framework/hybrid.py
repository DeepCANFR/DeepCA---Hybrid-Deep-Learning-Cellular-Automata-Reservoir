# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:03:01 2020

@author: trymlind
"""


import framework_module as fm
import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
Initiate model here
############################################################################################################
'''
# ex: neuron_poplulation_1 = fm.AS_Soma
population_x = 800
population_y = 800

neighbourhood_template = np.ones((3,3))
neighbourhood_template[1,1] = 0

neuron_population = fm.Soma_AS_with_projection_weights(0.1,0.5,0.5, population_x, population_y, neighbourhood_template, 1/(3*2),0.5)
conways_game_of_life = fm.Conways_game_of_life(population_x, population_y)

starting_state = np.zeros((population_x,population_y))
starting_state[400,400:450] = 1
starting_state[300:450,425] = 1
#starting_state = np.random.rand(population_x,population_y)>0.8
'''
############################################################################################################
'''

conways_game_of_life.input_method(starting_state, '+')

cap = cv2.VideoCapture(0)
video = np.zeros((2,2))
while True:
    
    '''
    Update model here
    ########################################################################################################
    '''
    # If using video input, uncomment below code. 
    # frame is the video input from the camera
    # ret, frame = cap.read()
    #conways_game_of_life.input_method(neuron_population.spiked_neurons,'XOR')
    video = np.concatenate((conways_game_of_life.universe, neuron_population.spiked_neurons),1)*255
    conways_game_of_life.compute()
    neuron_population.membrane_additive_input(conways_game_of_life.universe)
    neuron_population.compute_spikes()
    neuron_population.send_spikes_to_synapses(neuron_population.spiked_neurons)
    neuron_population.membrane_additive_input(np.sum(neuron_population.spike_array,2))
    #conways_game_of_life.input_method(neuron_population.spiked_neurons,'-')
    #conways_game_of_life.compute()
    
    
    
    
    # initialize video as the part you wish to visualize, ex neuron_population.spiked_neurons
    # video = output for visualiztion
    '''
    ########################################################################################################
    '''
    
    cv2.imshow('frame', video)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()