# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:45:36 2020

@author: trymlind

The module consists of two parts. One simulates the neurons used in the simulation and two provides methods for visualizing and measuring the neuronal activities
"""
import numpy as np
#import cupy as cp
import sys

DNA = {}
HOX_Gene = {}

class Organoid:
    def __init__(self, DNA):
        pass

def index(size):
    index = []
    for i0 in range(size):
        index.append(slice(None,None,1))

    return tuple(index)


class Synaptic_terminal:
    def __init__(self, inputs, neighbourhood_template):
        # The neighbourhood template should be a boolean 2d array where the central cell represents the position of a neuron and the neighbouring cells the position of the neighbours relative to the central cell. 1 signifies if the central is connected to the neighbour and 0 that they are not connected
        self.neighbourhood_template = neighbourhood_template
        print("Building synaptic terminal \n")
        if len(self.neighbourhood_template.shape) <= 1:
            print("Neihbourhood_template has 1 axis")
            self.template_rolls = [[0,0]]
            self.midX = 0
            self.midY = 0
            self.max_level = 1
            if len(inputs.shape) <= 1:
                print("Inputs has 1 axis of size: ", inputs.shape)
                self.spike_array = np.zeros(inputs.shape[0],dtype='float64')
            else:
                print("Inputs has 2 axis of size: ", inputs.shape)
                self.spike_array = np.zeros((inputs.shape[0], inputs.shape[1]) )
        else:
            print("Neihbourhood_template has 2 axis: \n ###################### \n", self.neighbourhood_template)
            print("######################")
            self.midX = int(-neighbourhood_template.shape[0]/2)
            self.midY = int(-neighbourhood_template.shape[1]/2)

            self.template_rolls = []
            self.max_level = 0
            for i0 in range(self.neighbourhood_template.shape[0]):
                for i1 in range(self.neighbourhood_template.shape[1]):
                    if self.neighbourhood_template[i0,i1] == 1:
                        self.template_rolls.append([self.midX + i0, self.midY + i1])
                        self.max_level += 1

            if len(inputs.shape) <= 1:
                print("Inputs have 1 axis of length: ", inputs.shape)
                self.spike_array = np.zeros((inputs.shape[0], self.max_level))
            elif (len(inputs.shape) == 2):
                print("Inputs have 2 axis of shape: ", inputs.shape)
                self.spike_array = np.zeros((inputs.shape[0], inputs.shape[1], self.max_level))
            else:
                print("######################### \n Error! \n #############################")
                print("Inputs have more than 2 axis: ", inputs.shape)
            # compute a list that gives the directions a spike should be sent to

            self.template_rolls = np.array(self.template_rolls)

    def send_spikes_to_synapses(self, inputs):
        print(inputs.shape)
        if self.max_level <= 1:
            self.spike_array = inputs
        else:
            for i0, x_y in enumerate(self.template_rolls):
                input_rolled = np.roll(inputs, x_y[0], axis = 0)
                input_rolled = np.roll(input_rolled, x_y[1], axis = 1)

                self.spike_array[:,:,i0] = input_rolled



class Preynaptic_terminal(Synaptic_terminal):
    def __init__(self, inputs, neighbourhood_template, max_neurotransmiter, release_ratio, cleft_re_uptake_rate, synaptic_gap_leak_rate, key):
        super().__init__(inputs, neighbourhood_template)



        self.neurotransmitter_reserve = np.ones(self.spike_array.shape, dtype='float64') * max_neurotransmiter
        self.neurotransmitter_synaptic_gap = np.zeros(self.spike_array.shape, dtype='float64')
        self.neurotransmitter_release = np.zeros(self.spike_array.shape, dtype='float64')
        self.neurotransmitter_ambient_diffusion = np.zeros(self.spike_array.shape, dtype='float64')
        self.neurotransmitter_move = np.zeros(self.spike_array.shape, dtype='float64')

        self.forward_output = np.full(self.neurotransmitter_synaptic_gap.shape, 0, dtype='float64')
        self.backward_output = None

        self.release_ratio = release_ratio
        self.cleft_re_uptake_rate = cleft_re_uptake_rate
        self.synaptic_gap_leak_rate = synaptic_gap_leak_rate

        if (self.cleft_re_uptake_rate + self.synaptic_gap_leak_rate) > 1:
            print("Error: reuptake rate and leak rate sum to greater than 1")
            sys.exit(0)


        self.key = key

    def compute_neurotransmitter_release(self):
        #print("spike array", self.spike_array.shape)
        #print("release", self.neurotransmitter_release.shape)
        self.neurotransmitter_release = self.spike_array * (self.neurotransmitter_reserve * self.release_ratio)

        self.neurotransmitter_synaptic_gap += self.neurotransmitter_release
        self.neurotransmitter_reserve -= self.neurotransmitter_release



    def compute_neurotransmitter_re_uptake(self):
        self.neurotransmitter_move = self.neurotransmitter_synaptic_gap * self.cleft_re_uptake_rate #(self.cleft_re_uptake_rate + self.synaptic_gap_leak_rate)

        self.neurotransmitter_reserve += self.neurotransmitter_move*self.cleft_re_uptake_rate
        self.neurotransmitter_synaptic_gap -= self.neurotransmitter_move*self.cleft_re_uptake_rate



    def transfer(self, forward_inputs, backward_inputs):
        self.send_spikes_to_synapses(forward_inputs)
        #print("neurotransmitter release", self.neurotransmitter_release)
        self.forward_output[index(len(self.forward_output.shape))] = self.neurotransmitter_release
        self.compute_neurotransmitter_re_uptake()

    def compute(self):
        self.compute_neurotransmitter_release()


class Postsynaptic_terminal:
    def __init__(self, inputs, lock):
        #Note that the spike array from the super class is used to store the feedback spikes
        self.terminal_membrane = np.full(inputs.shape,0, dtype='float64')
        if len(self.terminal_membrane.shape) <= 1:
            self.summed_inputs = np.full(1, 0, dtype='float64')
        else:
            self.summed_inputs = np.full(self.terminal_membrane.shape[0:-1], 0, dtype='float64')
        self.forward_output = np.full(self.summed_inputs.shape, 0, dtype='float64')
        self.backward_output = None
        self.lock = lock
        self.shape = self.terminal_membrane.shape

    def bind_neurontransmitter(self, inputs):

        self.terminal_membrane[:] = inputs
        #print(inputs, self.terminal_membrane)

    def sum_inputs(self):
        if len(self.shape) <= 1:
            self.summed_inputs[:] =  np.sum(self.terminal_membrane)
        else:
            self.summed_inputs[:,:] = np.sum(self.terminal_membrane,2)

    def transfer(self, forward_inputs, backward_inputs):

        #Transfer funcitons always have to compoents, one feedforward and one feedback
        if len(self.summed_inputs.shape) <= 1:
            self.forward_output[:] = self.summed_inputs
        else:
            self.forward_output[:,:] = self.summed_inputs
        #print("Postynaptic terminal transfer ", forward_inputs, self.forward_output )
        self.bind_neurontransmitter(forward_inputs)

        #return [self.forward_ouput, self.backward_output]


    def compute(self):
        self.sum_inputs()



class Postsynaptic_terminal_weighted(Postsynaptic_terminal):
    def __init__(self, inputs, weight_parameters, lock):
        #weight paramters should be a dict with parameters
        super().__init__(inputs, lock)
        #self.weighted_spikes = np.zeros(inputs.shape)


        try:
            weight_parameters["type"]
        except:
            print("type is missing")
            sys.exit(0)

        if weight_parameters["type"] == "random uniform":

                try:
                    self.weights = np.random.uniform(weight_parameters["low"], weight_parameters["high"], inputs.shape)
                except:
                    print("Parameters for weight type: uniform is missing")
                    sys.exit(0)

        elif (weight_parameters["type"] == "normal"):
            try:
                self.weights = np.random.normal(weight_parameters["mean"], weight_parameters["sd"], inputs.shape)
            except:
                print("Parameters missing for weight type: normal")
                sys.exit(0)
        elif (weight_parameters["type"] == "power"):
            try:
                self.weights = np.random.power(weight_parameters["a"], inputs.shape)
            except:
                print("Parameters missing for weight type: powe")
                sys.exit(0)


    def propagate_summed_weighted_spikes(self):
        return np.sum(self.weighted_spikes, 2)

    def compute(self):
        self.terminal_membrane *= self.weights
        self.sum_inputs()


class Postnynatpic_terminal_learning(Postsynaptic_terminal_weighted):
    def __init__(self, inputs, neighbourhood_template, lock, weight_parameters, soma_spike_memory_decay_rate, input_spike_memory_decay_rate):
        #weight paramters should be a dict with parapemeters
        self.super(inputs, neighbourhood_template, lock, weight_parameters)

        self.soma_spike_memory = np.zeros(self.weights.shape)
        self.input_spike_memory = np.zeros(self.weights.shape)

        self.input_spike_memory_decay_rate = input_spike_memory_decay_rate
        self.soma_spike_memory_decay_rate = soma_spike_memory_decay_rate



    def learn(self, somatic_feedback):
        #Somatic feedback is the spikes in the soma
        self.send_spikes_to_synapses(somatic_feedback)

        self.weights += ((self.soma.spike_memory - self.input_spike_memory_decay)/(self.weights + 10)) * self.spike_array

class Dendritic_compartment:
    def __init__(self, inputs, transfer_rate, leak_rate):
        self.compartment_membrane = np.zeros(inputs.shape)
        self.transfer_rate = transfer_rate
        self.leak_rate = leak_rate

    def inputs(self, inputs):
        self.compartment_membrane += inputs

    def transfer(self):

        self.compartment_membrane *= self.leak_rate
        self.transfer = self.compartment_membrane * self.transfer_rate
        self.compartment_membrane -= self.transfer
        return self.transfer

class Soma:
    def __init__(self):
        self.shape = None
        pass

    def compute(self):
        pass


class Soma_AS(Soma):
    # Model by Asgeir Bertland (AS)
    # source paper:
    def __init__(self, membrane_decay, treshold_decay, membrane_treshold_resting_distance, population_size_x, population_size_y):
        self.membrane_decay = membrane_decay
        self.threshold_decay = treshold_decay
        self.m_t_resting_distance = membrane_treshold_resting_distance

        self.membrane_potentials = np.zeros((population_size_x, population_size_y))
        self.spiked_membrane_potentials = np.zeros((population_size_x, population_size_y))
        self.unspiked_membrane_potentials = np.zeros((population_size_x, population_size_y))

        self.thresholds = np.ones((population_size_x, population_size_y))*self.m_t_resting_distance
        self.spiked_thresholds = np.zeros((population_size_x, population_size_y))
        self.unspiked_thresholds = np.zeros((population_size_x, population_size_y))

        self.spiked_neurons = np.zeros((population_size_x, population_size_y))
        self.unspiked_neurons = np.zeros((population_size_x, population_size_y))

        self.input_synapses = []
        self.shape = self.spiked_neurons.shape

        self.forward_output = np.full(self.spiked_neurons.shape, 0)
        self.backward_output = None

    def membrane_additive_input(self, inputs):
        # Adds inputs to membrane potential

        self.membrane_potentials[:,:] = self.membrane_potentials + inputs
        #print(self.membrane_potentials)
    def membrane_divisive_input(self, inputs):
        # Adds inputs to membrane potential
        self.membrane_potentials[:,:] = self.membrane_potentials * (1/inputs)

    def threshold_additive_input(self, inputs):
        self.thresholds[:,:] = self.thresholds + inputs

    def threshold_divisive_input(self, inputs):
        self.thresholds[:,:] = self.thresholds * (1/inputs)


    def compute_spikes(self):


        # Calculates which neurons spikes and which does not
        self.spiked_neurons[:,:] = self.membrane_potentials > self.thresholds
        self.unspiked_neurons[:,:] = self.spiked_neurons == 0

        # Calculates the new threshold value for all neurons that spiked
        self.spiked_thresholds[:,:] = (self.thresholds + self.threshold_decay * self.membrane_potentials)*self.spiked_neurons
        # Calclulates the thresholds of neurons that did not spike
        
        self.membrane_potentials[:,:] = self.membrane_potentials * self.membrane_decay * self.unspiked_neurons
        self.unspiked_thresholds[:,:] = (self.thresholds + self.threshold_decay * self.membrane_potentials) * self.unspiked_neurons

        self.thresholds = self.spiked_thresholds + self.unspiked_thresholds

        self.thresholds = self.thresholds + ((self.m_t_resting_distance - self.thresholds)/2)*self.threshold_decay


        

    def transfer(self, forward_inputs, backward_inputs):
        #print("Soma transfer:", forward_inputs)
        if len(self.shape) <= 1:
            self.forward_ouput[:] = self.spiked_neurons
            #print('x')
            self.membrane_additive_input(forward_inputs)
        else:
            #print('y')
            self.forward_output[:,:] = self.spiked_neurons
            self.membrane_additive_input(forward_inputs)


    def compute(self):
        self.compute_spikes()

class Soma_AS_with_projection_weights(Soma_AS):
        def __init__(self, membrane_decay, treshold_decay, membrane_treshold_resting_distance, population_size_x, population_size_y, neighbourhood_template, weight_mean, weight_SD):
            # The neighbourhood template should be a boolean 2d array where the central cell represents the position of a neuron and the neighbouring cells the position of the neighbours relative to the central cell. 1 signifies if the central is connected to the neighbour and 0 that they are not connected
            super().__init__(membrane_decay, treshold_decay, membrane_treshold_resting_distance, population_size_x, population_size_y)
            inputs = self.spiked_neurons
            
            self.neighbourhood_template = neighbourhood_template
            print("Building synaptic terminal \n")
            if len(self.neighbourhood_template.shape) <= 1:
                print("Neihbourhood_template has 1 axis")
                self.template_rolls = [[0,0]]
                self.midX = 0
                self.midY = 0
                self.max_level = 1
                if len(inputs.shape) <= 1:
                    print("Inputs has 1 axis of size: ", inputs.shape)
                    self.spike_array = np.zeros(inputs.shape[0],dtype='float64')
                else:
                    print("Inputs has 2 axis of size: ", inputs.shape)
                    self.spike_array = np.zeros((inputs.shape[0], inputs.shape[1]) )
            else:
                print("Neihbourhood_template has 2 axis: \n ###################### \n", self.neighbourhood_template)
                print("######################")
                self.midX = int(-neighbourhood_template.shape[0]/2)
                self.midY = int(-neighbourhood_template.shape[1]/2)
    
                self.template_rolls = []
                self.max_level = 0
                for i0 in range(self.neighbourhood_template.shape[0]):
                    for i1 in range(self.neighbourhood_template.shape[1]):
                        if self.neighbourhood_template[i0,i1] == 1:
                            self.template_rolls.append([self.midX + i0, self.midY + i1])
                            self.max_level += 1
    
                if len(inputs.shape) <= 1:
                    print("Inputs have 1 axis of length: ", inputs.shape)
                    self.spike_array = np.zeros((inputs.shape[0], self.max_level))
                elif (len(inputs.shape) == 2):
                    print("Inputs have 2 axis of shape: ", inputs.shape)
                    self.spike_array = np.zeros((inputs.shape[0], inputs.shape[1], self.max_level))
                else:
                    print("######################### \n Error! \n #############################")
                    print("Inputs have more than 2 axis: ", inputs.shape)
                # compute a list that gives the directions a spike should be sent to
    
                self.template_rolls = np.array(self.template_rolls)
    
                self.weights = np.abs(np.random.normal(weight_mean, weight_SD, self.spike_array.shape))
        def send_spikes_to_synapses(self, inputs):
            print(inputs.shape)
            if self.max_level <= 1:
                self.spike_array = inputs
            else:
                for i0, x_y in enumerate(self.template_rolls):
                    input_rolled = np.roll(inputs, x_y[0], axis = 0)
                    input_rolled = np.roll(input_rolled, x_y[1], axis = 1)
    
                    self.spike_array[:,:,i0] = input_rolled
    
            self.spike_array *= self.weights

'''
##################################################################################################################################################################
CELLULAR AUTOMATA
'''


class Automata_1d:
    def __init__(self, universe_in_starting_state, rule):
        self.universe = np.uint64(universe_in_starting_state)
        
        universe_size = len(self.universe)
        
        self.states = list(rule.keys())
        self.neighbourhoodsize = len(self.states[0])
        self.neighbourhoodstate = np.zeros((universe_size, self.neighbourhoodsize), dtype = np.uint64)
        
        self.nr_of_transitions_to_1 = 0
        for key in rule:
            self.nr_of_transitions_to_1 += rule[key]
        
        self.rule_matrix = np.zeros((universe_size, self.neighbourhoodsize, self.nr_of_transitions_to_1), dtype = np.uint64)
        self.rule_state_matrix = np.zeros((universe_size, self.neighbourhoodsize, self.nr_of_transitions_to_1), dtype = np.uint64)
        
        j = 0
        for i0, key in enumerate(rule):
            if rule[key] == 1:
                for i1 in range(len(key)):
                    self.rule_matrix[:, i1, j] += int(key[i1])
                j += 1
                
        
    def do_neighbourhood(self):
        
        for i0 in range(self.neighbourhoodsize):
            self.neighbourhoodstate[:,i0] = np.roll(self.universe, i0-1)
            
    def update_universe(self):
        self.do_neighbourhood()
        
        for i0 in range(self.nr_of_transitions_to_1):
        
            self.rule_state_matrix[:,:,i0] = self.neighbourhoodstate == self.rule_matrix[:,:,i0]
        self.universe = np.sum(np.prod(self.rule_state_matrix,1),1)
        
    def kill_cell(self, index):
        self.rule_matrix[index,:,:] +=2
    
class Automata_2d:
    def __init__(self, size_x, size_y, neighbourhood_template):
            self.universe = np.zeros((size_x, size_y))
            
            
            self.neighbourhood_template = neighbourhood_template
            
            self.midX = int(-neighbourhood_template.shape[0]/2)
            self.midY = int(-neighbourhood_template.shape[1]/2)

            self.template_rolls = []
            self.max_level = 0
            for i0 in range(self.neighbourhood_template.shape[0]):
                for i1 in range(self.neighbourhood_template.shape[1]):
                    if self.neighbourhood_template[i0,i1] == 1:
                        self.template_rolls.append([self.midX + i0, self.midY + i1])
                        self.max_level += 1
            self.template_rolls = np.array(self.template_rolls)
            
            self.neighbourhood = np.zeros((size_x, size_y, self.max_level))
            
    def send_state_to_neighbourhood(self):
            
      
        for i0, x_y in enumerate(self.template_rolls):
            input_rolled = np.roll(self.universe, x_y[0], axis = 0)
            input_rolled = np.roll(input_rolled, x_y[1], axis = 1)

            self.neighbourhood[:,:,i0] = input_rolled

class Conways_game_of_life(Automata_2d):
    def __init__(self, size_x, size_y):
        neighbourhood_template = np.ones((3,3))
        neighbourhood_template[1,1] = 0
        super().__init__(size_x, size_y, neighbourhood_template)
        
    def input_method(self, inputs, input_type):
        if input_type == "+":
            self.universe = (self.universe + inputs)
            self.universe = self.universe >= 1
        elif (input_type == "-"):
            self.universe = (self.universe - inputs)
            self.universe = self.universe == 1
        elif (input_type == "XOR"):
            self.universe = self.universe + inputs
            self.universe = self.universe == 1
            
        
    def compute(self, inputs = 0, input_type = "XOR"):
        self.input_method(inputs, input_type)
        
        self.send_state_to_neighbourhood()
        
        neighbourhood_sum = np.sum(self.neighbourhood,2)
        print(neighbourhood_sum)
        
        # we only need to compute the rules that leads to living cells
        # rule 2
        # aAny live cell with two or three live neighbours lives on to the next generation
        rule_2 = self.universe * ((neighbourhood_sum > 1) * (neighbourhood_sum < 4))
        
        rule_4 = (self.universe == 0) * (neighbourhood_sum == 3)
        
        self.universe = rule_2 + rule_4
        
    
class MEA:
    '''
    Micro Electrode Array (MEA)
    '''
    def __init__(self):
        pass

    def measure(self):
        pass
    def stimulate(self):
        pass

class Recorder:
    def __init__(self, MEA):
        self.MEA = MEA
    def load_stimulation_pattern(self, stimulation_pattern):
        pass
