# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:14:08 2021

@author: trymlind
"""


import realistic_module_test as rm
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
import multiprocessing
import concurrent.futures
import sys

try:
    import cupy as cp
    cupy = True
except:
    print("Failed to import cupy, attempting to import numpy instead")
    cupy = False
    import numpy as cp
    
import numpy as np
import cv2

'''
Load stimuli
#######################################################################################
'''
(image_train, label_train), (image_test, label_test) = tf.keras.datasets.mnist.load_data()


'''
Set parameters
'''

weight_scaling = 5

time_step = 0.1
population_size = image_train[0,:,:].shape


excitatory_soma_parameters = {}
excitatory_soma_parameters["population_size"] = population_size
excitatory_soma_parameters["membrane_time_constant"] = 30 # ms
excitatory_soma_parameters["absolute_refractory_period"] = 3 # ms
excitatory_soma_parameters["threshold"] = 15 # mv
excitatory_soma_parameters["reset_voltage"] = 13.5 # mv
excitatory_soma_parameters["background_current"] = 13.5 # nA
excitatory_soma_parameters["input_resistance"] = 1 # M_Ohm
excitatory_soma_parameters["refractory_period"] = 3
excitatory_soma_parameters["time_step"] = time_step

inhibitory_soma_parameters = {}
inhibitory_soma_parameters["population_size"] = population_size
inhibitory_soma_parameters["membrane_time_constant"] = 30 # ms
inhibitory_soma_parameters["absolute_refractory_period"] = 2 # ms
inhibitory_soma_parameters["threshold"] = 15 # mv
inhibitory_soma_parameters["reset_voltage"] = 13.5 # mv
inhibitory_soma_parameters["background_current"] = 13.5 # nA
inhibitory_soma_parameters["input_resistance"] = 1 # M_Ohm
inhibitory_soma_parameters["refractory_period"] = 2
inhibitory_soma_parameters["time_step"] = time_step

# to do: make all paramets only have positive values
excitatory_to_excitatory_dynamical_synapse_parameters = {}
excitatory_to_excitatory_dynamical_synapse_parameters["resting_utilization_of_synaptic_efficacy"] = {"distribution":"normal", "mean":0.5, "SD":0.5/2}# U 
excitatory_to_excitatory_dynamical_synapse_parameters["time_constant_depresssion"] = {"distribution":"normal", "mean":1.1, "SD":1.1/2}# in Maas et al: D, in Markram et al: tau_depression  expressed in seconds
excitatory_to_excitatory_dynamical_synapse_parameters["time_constant_facilitation"] = {"distribution":"normal", "mean":0.5, "SD":0.5/2}# in Maas et al: F, in Markram et al: tau_facil
# to do: this should be from a gamma distribution I think, but I don't understand how they made it
excitatory_to_excitatory_dynamical_synapse_parameters["absolute_synaptic_efficacy"] = {"distribution":"normal", "mean":30*weight_scaling, "SD":30*weight_scaling}# in Maas et al: A, in Markram et al: A
excitatory_to_excitatory_dynamical_synapse_parameters["type"] = "excitatory"
excitatory_to_excitatory_dynamical_synapse_parameters["time_step"] = time_step

excitatory_to_inhibitory_dynamical_synapse_parameters = {}
excitatory_to_inhibitory_dynamical_synapse_parameters["resting_utilization_of_synaptic_efficacy"] = {"distribution":"normal", "mean":0.5, "SD":0.5/2}# U  Strange, the setting from the paper is 0.05, but this results in the inhibitory neurons not firing
excitatory_to_inhibitory_dynamical_synapse_parameters["time_constant_depresssion"] = {"distribution":"normal", "mean":0.125, "SD":0.125/2} # in Maas et al: D, in Markram et al: tau_depression  expressed in seconds
excitatory_to_inhibitory_dynamical_synapse_parameters["time_constant_facilitation"] = {"distribution":"normal", "mean":1.2, "SD":1.2/2} # in Maas et al: F, in Markram et al: tau_facil
# to do: this should be from a gamma distribution I think, but I don't understand how they made it
excitatory_to_inhibitory_dynamical_synapse_parameters["absolute_synaptic_efficacy"] = {"distribution":"normal", "mean":60*weight_scaling, "SD":60*weight_scaling}# in Maas et al: A, in Markram et al: A
excitatory_to_inhibitory_dynamical_synapse_parameters["type"] = "excitatory"
excitatory_to_inhibitory_dynamical_synapse_parameters["time_step"] = time_step

inhibitory_to_excitatory_dynamical_synapse_parameters = {}
inhibitory_to_excitatory_dynamical_synapse_parameters["resting_utilization_of_synaptic_efficacy"] = {"distribution":"normal", "mean":0.25, "SD":0.25/2} # U 
inhibitory_to_excitatory_dynamical_synapse_parameters["time_constant_depresssion"] = {"distribution":"normal", "mean":0.7, "SD":0.7/2} # in Maas et al: D, in Markram et al: tau_depression  expressed in seconds
inhibitory_to_excitatory_dynamical_synapse_parameters["time_constant_facilitation"] = {"distribution":"normal", "mean":0.02, "SD":0.02/2} # in Maas et al: F, in Markram et al: tau_facil
# to do: this should be from a gamma distribution I think, but I don't understand how they made it
inhibitory_to_excitatory_dynamical_synapse_parameters["absolute_synaptic_efficacy"] = {"distribution":"normal", "mean":-19*weight_scaling, "SD":19*weight_scaling}# in Maas et al: A, in Markram et al: A
inhibitory_to_excitatory_dynamical_synapse_parameters["type"] = "inhibitory"
inhibitory_to_excitatory_dynamical_synapse_parameters["time_step"] = time_step

inhibitory_to_inhibitory_dynamical_synapse_parameters = {}
inhibitory_to_inhibitory_dynamical_synapse_parameters["resting_utilization_of_synaptic_efficacy"] = {"distribution":"normal", "mean":0.32, "SD":0.32/2} # U 
inhibitory_to_inhibitory_dynamical_synapse_parameters["time_constant_depresssion"] = {"distribution":"normal", "mean":0.144, "SD":0.144/2} # in Maas et al: D, in Markram et al: tau_depression  expressed in seconds
inhibitory_to_inhibitory_dynamical_synapse_parameters["time_constant_facilitation"] = {"distribution":"normal", "mean":0.06, "SD":0.06/2} # in Maas et al: F, in Markram et al: tau_facil
# to do: this should be from a gamma distribution I think, but I don't understand how they made it
inhibitory_to_inhibitory_dynamical_synapse_parameters["absolute_synaptic_efficacy"] = {"distribution":"normal", "mean":-19*weight_scaling, "SD":19*weight_scaling}# in Maas et al: A, in Markram et al: A
inhibitory_to_inhibitory_dynamical_synapse_parameters["type"] = "inhibitory"
inhibitory_to_inhibitory_dynamical_synapse_parameters["time_step"] = time_step


EE_dendritic_arbor_parameters = {}
EE_dendritic_arbor_parameters["projection_template"] = cp.ones((7,7))
EE_dendritic_arbor_parameters["time_step"] = time_step
EE_dendritic_arbor_parameters["distance_based_connection_probability"] = {"C":0.3, "lambda_parameter":5}

EE_delay_line_parameters = {}
EE_delay_line_parameters["delay"] = 1.5 # ms
EE_delay_line_parameters["time_step"] = time_step #ms

EI_dendritic_arbor_parameters = {}
EI_dendritic_arbor_parameters["projection_template"] = cp.ones((7,7))
EI_dendritic_arbor_parameters["projection_template"][1:5,1:5] = 0
EI_dendritic_arbor_parameters["time_step"] = time_step
EI_dendritic_arbor_parameters["distance_based_connection_probability"] = {"C":0.2, "lambda_parameter":5}

EI_delay_line_parameters = {}
EI_delay_line_parameters["delay"] = 0.8
EI_delay_line_parameters["time_step"] = time_step

IE_dendritic_arbor_parameters = {}
IE_dendritic_arbor_parameters["projection_template"] = cp.ones((5,5))
IE_dendritic_arbor_parameters["time_step"] = time_step
IE_dendritic_arbor_parameters["distance_based_connection_probability"] = {"C":0.4, "lambda_parameter":5}

#excitatory dendritic spine
E_dendritic_Spine_parameters = {}
E_dendritic_Spine_parameters["time_step"] = time_step
E_dendritic_Spine_parameters["time_constant"] = 3 # ms

#inhibitory dendritic spine
I_dendritic_Spine_parameters = {}
I_dendritic_Spine_parameters["time_step"] = time_step
I_dendritic_Spine_parameters["time_constant"] = 6 # ms

IE_delay_line_parameters = {}
IE_delay_line_parameters["delay"] = 0.8
IE_delay_line_parameters["time_step"] = time_step



P_delta_readout_parameters = {}
P_delta_readout_parameters["nr_of_readout_neurons"] = 51
P_delta_readout_parameters["error_tolerance"] = 0.05
P_delta_readout_parameters["rho"] = 1 # squashing function boundries
P_delta_readout_parameters["margin"] = 0.02
P_delta_readout_parameters["clear_margins_importance"] = 1
P_delta_readout_parameters["learning_rate"] = 0.0025

'''
Build neuron
######################################################################################
'''

# Initiate components
class Layer(object):
    def __init__(self):
        
        self.excitatory_somas = rm.Simple_Integrate_and_fire_soma(excitatory_soma_parameters)
        
        self.inhibitory_somas = rm.Simple_Integrate_and_fire_soma(inhibitory_soma_parameters)
        self.inhibitory_somas.set_dead_cells(cp.random.uniform(0,1,population_size)<0.2)
        
        self.EE_delay_line = rm.Delay_Line(EE_delay_line_parameters)
        self.EE_dendritic_arbor = rm.Dendritic_Arbor(EE_dendritic_arbor_parameters)
        self.EE_axonal_terminal = rm.Dynamical_Axonal_Terminal_Markram_etal_1998(excitatory_to_excitatory_dynamical_synapse_parameters)
        self.EE_dendritic_spine = rm.Dendritic_Spine_Maas(E_dendritic_Spine_parameters)
        
        self.EI_delay_line = rm.Delay_Line(EI_delay_line_parameters)
        self.EI_dendritic_arbor = rm.Dendritic_Arbor(EI_dendritic_arbor_parameters)
        self.EI_axonal_terminal = rm.Dynamical_Axonal_Terminal_Markram_etal_1998(excitatory_to_inhibitory_dynamical_synapse_parameters)
        self.EI_dendritic_spine = rm.Dendritic_Spine_Maas(E_dendritic_Spine_parameters)
        
        self.IE_delay_line = rm.Delay_Line(IE_delay_line_parameters)
        self.IE_dendritic_arbor = rm.Dendritic_Arbor(IE_dendritic_arbor_parameters)
        self.IE_axonal_terminal = rm.Dynamical_Axonal_Terminal_Markram_etal_1998(inhibitory_to_excitatory_dynamical_synapse_parameters)
        self.IE_dendritic_spine = rm.Dendritic_Spine_Maas(E_dendritic_Spine_parameters)
        
        
        # connect components
        self.EE_delay_line.interface_read_variable(self.excitatory_somas.current_spiked_neurons)
        self.EE_dendritic_arbor.interface(self.EE_delay_line.current_spike_output)
        self.EE_axonal_terminal.interface_read_variable(self.EE_dendritic_arbor.current_spike_array)
        
        
        self.EI_delay_line.interface_read_variable(self.excitatory_somas.current_spiked_neurons)
        self.EI_dendritic_arbor.interface(self.EI_delay_line.current_spike_output)
        self.EI_axonal_terminal.interface_read_variable(self.EI_dendritic_arbor.current_spike_array)
        self.EI_dendritic_spine.interface_read_variable(self.EI_axonal_terminal.current_synaptic_response)
        self.inhibitory_somas.interface_input(self.EI_dendritic_spine.current_synaptic_output)
        
        self.IE_delay_line.interface_read_variable(self.inhibitory_somas.current_spiked_neurons)
        self.IE_dendritic_arbor.interface(self.IE_delay_line.current_spike_output)
        self.IE_axonal_terminal.interface_read_variable(self.IE_dendritic_arbor.current_spike_array)
        self.IE_dendritic_spine.interface_read_variable(self.IE_axonal_terminal.current_synaptic_response)
        self.excitatory_somas.interface_input(self.IE_dendritic_spine.current_synaptic_output)
        
        self.excitatory_somas.interface_input(background_current)
        self.inhibitory_somas.interface_input(background_current)
        
        self.component_list = [self.EE_dendritic_arbor, self.EE_delay_line, self.EE_axonal_terminal, self.excitatory_somas, self.inhibitory_somas, self.EI_dendritic_arbor, self.EI_axonal_terminal, self.EI_delay_line, self.IE_dendritic_arbor, self.IE_axonal_terminal, self.IE_delay_line, self.EE_dendritic_spine, self.EI_dendritic_spine, self.IE_dendritic_spine]
        
    def compute_new_values(self):
        for component in self.component_list:
            component.compute_new_values()
    def update_current_values(self):
        for component in self.component_list:
            component.update_current_values()
            
background_current = cp.ones(population_size)*13.5
background_current = background_current[:,:,cp.newaxis]

stimulation_input = cp.zeros(population_size)
stimulation_input = stimulation_input[:,:,cp.newaxis]

layer_1 = Layer()
layer_2 = Layer()
layer_3 = Layer()
layer_4 = Layer()
layer_5 = Layer()
layer_6 = Layer()
layer_7 = Layer()

network = [layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7]

for index, layer in enumerate(network):
    if index == 0:
        layer.EE_dendritic_spine.interface_read_variable(stimulation_input)
        layer.excitatory_somas.interface_input(layer.EE_dendritic_spine.current_synaptic_output)
        previous_layer = layer
    else:
        layer.EE_dendritic_spine.interface_read_variable(previous_layer.EE_axonal_terminal.current_synaptic_response)
        layer.excitatory_somas.interface_input(layer.EE_dendritic_spine.current_synaptic_output)
        previous_layer = layer
'''
Build network
######################################################################################
'''





'''
Initialize readout mechanism
######################################################################################
'''
readout_list = []
for class_nr in range(10):
    readout_list.append(rm.Readout_P_Delta(P_delta_readout_parameters))
    readout_list[class_nr].interface_read_variable(layer_7.excitatory_somas.current_spiked_neurons)





'''
Set simulation parameters
######################################################################################
'''

nr_of_epochs = 100
timesteps_per_image = int(100/time_step)

'''
Output animation
######################################################################################
'''

trial_guesses = np.empty((10, timesteps_per_image))
trial_guesses[:,:] = 0



'''
Run simulation
######################################################################################
'''



t = 0

stop_simulation = False


'''
Simulation starts here, after each training epoch the accuracy of the network is tested
By pressing 'q' on the keyboad during the training phase you can skip to the testing phase.
If you again press 'q' during the testing phase you will end the simulation
'''
for epoch in range(nr_of_epochs):
    '''
    Training ########################################################################
    '''
    print("")
    print("########## - Training phase - #########")
    print("Starting epoch nr: ", epoch)
    
    mode_correct_classifications = 0
    mean_correct_classifications = 0
    plt.ion()
    #fig = plt.figure()
    #ax1 = plt.axes(xlin = (0,time_window), ylin = (-1.5, 1.5))
    #line, = ax1.plot([],[], lw = 2)
    #plt.title("Readout activity")
    lines = plt.plot(np.nanmean(trial_guesses, axis = 1), '*')
    plt.axis([0,10, -1.5,1.5])
    
    for image_nr in range(len(image_train)):
        current_guesses = np.zeros(10)
        trial_guesses = np.empty((10, timesteps_per_image))
        trial_guesses[:,:] = np.nan
        
        print("")
        print("Training: current_class: ", label_train[image_nr])
        stimulation_input[:,:,0] = (cp.array(image_train[image_nr,:,:])/255)*120
        
        for trial_time_step in range(timesteps_per_image):
            t += time_step
            
            print("Trial time: ", np.round((trial_time_step + 1)*time_step, 2), " ms", end = '\r')
            
            
           
            for layer in network:
                layer.compute_new_values()
            '''
            print("")
            print(__name__)
            if __name__ == '__main__x':
                print("parall")
                network_compute_new_processes = []
                network_update_current_values_processes = []
                
                for component in network:
                    network_compute_new_processes.append(multiprocessing.Process(target = component.compute_new_values))
                    network_update_current_values_processes.append(multiprocessing.Process(target = component.update_current_values))

                for component in network_compute_new_processes:
                    component.start()
                for component in network_compute_new_processes:
                    component.join()
            else:
                print("failed to run parallel code")
                sys.exit(0)
            '''
          
            for layer in network:
                layer.update_current_values()
            #print(excitatory_somas.current_somatic_voltages[51,50])
            
            if cp.any(layer_7.excitatory_somas.current_spiked_neurons > 0) and (trial_time_step+1)*time_step > 20:
                for class_nr, readout in enumerate(readout_list):
                    
                    # Find if the current readout corresponds with the current stimulation label and computes the corresponding desired output
                    desired_output = label_train[image_nr] == class_nr
                    if desired_output == 0:
                        desired_output = -1
                        
                    # update weights given desired output
                    readout.update_weights(desired_output)
                    
                    current_guesses[class_nr] = readout.current_population_output
                
                trial_guesses[:, trial_time_step] = current_guesses
             
            lines[0].set_ydata(np.nanmean(trial_guesses, axis = 1))
            plt.show()
            
            image = image_train[image_nr,:,:]
            if cupy:
                #image= cp.asnumpy(excitatory_somas.current_somatic_voltages)
                #image= cp.asnumpy(inhibitory_somas.current_somatic_voltages)
                for layer in network:
                    activity = cp.asnumpy(layer.excitatory_somas.current_spiked_neurons)
                    image = np.concatenate((image,activity), axis = 1)
                
                #image= cp.asnumpy(inhibitory_somas.current_spiked_neurons)
                #image = cp.asnumpy(cp.sum(readout.weights, axis = 2))
            else:
                image= layer_7.excitatory_somas.current_somatic_voltages*255
            
           
            image_shape = np.array(image.shape)
            image_shape *= 10
            image_shape = image_shape[::-1]
            image_shape = tuple(image_shape)
            image = cv2.resize(image,image_shape)
            cv2.imshow('frame', image)

         
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_simulation = True
                break
        if stop_simulation:
                break
            
        # compute what the outputs most frequent guess was
        classifications_during_trial = np.argmax(trial_guesses, axis = 0)
        #print(classifications_during_trial)
        mode_guess, count = stats.mode(classifications_during_trial, axis = None)
        mode_correct_classifications += int(mode_guess[0]) == label_train[image_nr]
        
        mean_guess = np.nanmean(trial_guesses, axis = 1)
        mean_guess = np.argmax(mean_guess)
        mean_correct_classifications += int(mean_guess) == label_train[image_nr]
      
        print("")
        print("Current mode guess: ", int(mode_guess[0]))
        print("Current mode accuracy; ",mode_correct_classifications/(image_nr + 1))
        print("")
        print("Current mean guess: ", int(mean_guess))
        print("Current mean accuracy; ", mean_correct_classifications/(image_nr + 1))
    stop_simulation = False
    
    
    '''
    Testing #########################################################################
    '''
    print("")
    print("########## - Starting Testing - #########")
          
    plt.ion()
    lines = plt.plot(np.nanmean(trial_guesses, axis = 1), '*')
    plt.axis([0,10, -1.5,1.5])
    
    # Test accuracy on test data
    mode_correct_classifications = 0
    mean_correct_classifications = 0
    t = 0
    guesses_during_trial = np.zeros(timesteps_per_image)
    current_guesses = np.zeros(10)
    for image_nr in range(len(image_test)):
        guesses_during_trial = cp.zeros(timesteps_per_image)
        
        current_guesses = np.zeros(10)
        trial_guesses = np.empty((10, timesteps_per_image))
        trial_guesses[:,:] = np.nan
        
        print("")
        print("Test: current_class: ", label_test[image_nr])
        
        stimulation_input[:,:,0] = (cp.array(image_test[image_nr,:,:])/255)*18
        
        for image_time_step in range(timesteps_per_image):
            t += time_step
            
            print("Trial time: ", np.round((trial_time_step + 1)*time_step, 2), " ms", end = '\r')
            
            
            
            
            for component in network:
                component.compute_new_values()
            
           
            for component in network:
                component.update_current_values()
            #print(excitatory_somas.current_somatic_voltages[51,50])
            
            if cp.any(excitatory_somas.current_spiked_neurons > 0):
                for class_nr, readout in enumerate(readout_list):
                
                    current_guesses[class_nr] = readout.classify(excitatory_somas.current_spiked_neurons)
                    
               
            trial_guesses[:, image_time_step] = current_guesses
            
            lines[0].set_ydata(np.nanmean(trial_guesses,axis = 1))
            #print(np.mean(output_plot_frame[:,0:500],axis = 1))
            plt.show()
            if cupy:
                #image= cp.asnumpy(excitatory_somas.current_somatic_voltages)
                #image= cp.asnumpy(inhibitory_somas.current_somatic_voltages)
                image= cp.asnumpy(excitatory_somas.current_spiked_neurons)
                #image= cp.asnumpy(inhibitory_somas.current_spiked_neurons)
                #image = cp.asnumpy(cp.sum(readout.weights, axis = 2))
            else:
                image= excitatory_somas.current_somatic_voltages*255
            
            image = np.concatenate((image, image_test[image_nr,:,:]), axis = 1)
            image_shape = np.array(image.shape)
            image_shape *= 10
            image_shape = image_shape[::-1]
            image_shape = tuple(image_shape)
            image = cv2.resize(image,image_shape)
            cv2.imshow('frame', image)

         
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_simulation = True
                break
            
        # compute what the outputs most frequent guess was
        classifications_during_trial = np.argmax(trial_guesses, axis = 0)
        #print(classifications_during_trial)
        mode_guess, count = stats.mode(classifications_during_trial, axis = None)
        mode_correct_classifications += int(mode_guess[0]) == label_test[image_nr]
        
        mean_guess = np.nanmean(trial_guesses, axis = 1)
        mean_guess = np.argmax(mean_guess)
        mean_correct_classifications += int(mean_guess) == label_test[image_nr]
      
        print("")
        print("Current mode guess: ", int(mode_guess[0]))
        print("Current mode accuracy; ",mode_correct_classifications/(image_nr + 1))
        print("")
        print("Current mean guess: ", int(mean_guess))
        print("Current mean accuracy; ", mean_correct_classifications/(image_nr + 1))
        
        
        if stop_simulation:
                break
    if stop_simulation:
                break

   
    
    

cv2.destroyAllWindows()

