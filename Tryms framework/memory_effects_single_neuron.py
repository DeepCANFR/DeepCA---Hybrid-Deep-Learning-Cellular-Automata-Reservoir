# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:31:31 2020

@author: trymlind
"""


import framework_module as fm
import numpy as np
import matplotlib.pyplot as plt

sim_length = 100
input_pattern_shape = (2,sim_length)


input_pattern_1 = np.zeros(input_pattern_shape)
input_pattern_1[0,20:22] = 0.1
input_pattern_1[0,70:75] = 0.5


class Static_Synapse_Neuron:
    
    def __init__(self, membrane_decay, threshold_decay, resting_difference):
        '''
        All components of must be placed in a list in an order corresponding to information flow
        '''
        self.soma = fm.Soma_AS(membrane_decay, threshold_decay, resting_difference,1,1)
        self.postsynaptic_terminal = fm.Postsynaptic_terminal(np.zeros(2), lock = "1")
        self.components = [self.soma, self.postsynaptic_terminal]
        self.forward_output = np.full(self.components[0].shape,0)
        
    def transfer(self, inputs):
        print('\n Static synapse transfer')
        self.forward_output = self.components[0].transfer(self.components[1].forward_output, self.components[1].backward_output)
        for i0 in range(1,len(self.components)-1):
            print(i0)
            self.components[i0].transfer(self.components[i0+1].forward_output, self.components[i0+1].backward_output)
    
        
        self.components[-1].transfer(inputs, backward_inputs = None)
        
    def compute(self):
        for i0 in range(len(self.components)):
            self.components[i0].compute()

class Dynamic_Synapse_Neuron(Static_Synapse_Neuron):
    def __init__(self):
        super().__init__()
        inputs = np.zeros(2)
        print(inputs)
        self.components.append(fm.Preynaptic_terminal(inputs, neighbourhood_template = np.array(1), max_neurotransmiter = 2, release_ratio = 0.5, cleft_re_uptake_rate = 0.1, synaptic_gap_leak_rate = 0, key = '1'))
        
    
    
        
static_synapse_neuron_fast_memory = Static_Synapse_Neuron(0.3, 0.9, 0.5)
static_synapse_neuron_slow_memory = Static_Synapse_Neuron(0.99, 0.9, 0.5)

fast_memory_history_membrane = np.zeros(sim_length)
fast_memory_history_threshold = np.zeros(sim_length)
fast_memory_history_spike = np.zeros(sim_length)

slow_memory_history_membrane = np.zeros(sim_length)
slow_memory_history_threshold = np.zeros(sim_length)
slow_memory_history_spike = np.zeros(sim_length)

for i0 in range(sim_length):
    static_synapse_neuron_fast_memory.compute()
    static_synapse_neuron_fast_memory.transfer(input_pattern_1[:,i0])
    
    static_synapse_neuron_slow_memory.compute()
    static_synapse_neuron_slow_memory.transfer(input_pattern_1[:,i0])
    
    
    fast_memory_history_membrane[i0] = static_synapse_neuron_fast_memory.soma.membrane_potentials
    fast_memory_history_spike[i0] = static_synapse_neuron_fast_memory.soma.spiked_neurons
    fast_memory_history_threshold[i0] = static_synapse_neuron_fast_memory.soma.thresholds
    
    slow_memory_history_membrane[i0] = static_synapse_neuron_slow_memory.soma.membrane_potentials
    slow_memory_history_spike[i0] = static_synapse_neuron_slow_memory.soma.spiked_neurons
    slow_memory_history_threshold[i0] = static_synapse_neuron_slow_memory.soma.thresholds
    '''
    
    static_history_membrane[i0] = static_synapse_neuron.postsynaptic_terminal.summed_inputs
    static_history_spike[i0] = static_synapse_neuron.postsynaptic_terminal.terminal_membrane[0]
    static_history_threshold[i0] = static_synapse_neuron.postsynaptic_terminal.forward_output
    #static_history[i0] = static_synapse_neuron.postsynaptic_terminal.forward_output
    '''
plt.figure()
plt.subplot(2,1,1)
#plt.figure("Dynamic synapse")
plt.plot(fast_memory_history_membrane, label = "Membrane")
plt.plot(fast_memory_history_threshold, label = "Threshold")
plt.plot(fast_memory_history_spike, '.', label = "Spikes")
plt.plot(input_pattern_1[0,:], label = "Input channel 1", linestyle = ':')
plt.plot(input_pattern_1[1,:], label = "Input channel 2", linestyle = ':')


   
plt.subplot(2,1,2)
#plt.title("Static synapse")
plt.plot(slow_memory_history_membrane, label = "Membrane")
plt.plot(slow_memory_history_threshold, label = "Threshold")
plt.plot(slow_memory_history_spike, '.', label = "Spikes")
plt.plot(input_pattern_1[0,:], label = "Input channel 1", linestyle = ':')
plt.plot(input_pattern_1[1,:], label = "Input channel 2", linestyle = ':')
plt.legend()
plt.show()
    