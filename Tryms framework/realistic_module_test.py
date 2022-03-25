# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:26:11 2020

@author: trymlind
"""

try:
    import cupy as cp
    cupy = True
except Exception:
    print("Failed to import cupy, attempting to import numpy instead")
    cupy = False
    import numpy as cp

import dask as dask
import sys
import cv2

upper_limit = 100000

class RungeKutta2_cupy(object):
    def __init__(self, f, time_step):
        # wraps user's f in a new function that always
        # converts lists/tuples to array (or let arrays be arrays)
        #self.f = lambda u,t: cp.asarray(f(u,t), float)
        self.f = f
        self.time_step = time_step # size of time step
        self.t = 0

    def advance(self, u, t):

        K1 = self.time_step * (self.f(u,t))
        K2 = self.time_step * self.f(u + (1/2)* K1, t + (1/2)*self.time_step)



        t += self.time_step
        u_delta = K2

        return u_delta

    def set_initial_condition(self, U0):
        if isinstance(U0, (float,int)):
            self.neq = 1
            U0 = float(U0)
        else:
            U0 = cp.asarray(U0)
            self.neq = U0.size

        self.U0 = U0

class ForwardEuler_cupy(object):
    def __init__(self, f, time_step):
        # wraps user's f in a new function that always
        # converts lists/tuples to array (or let arrays be arrays)
        #self.f = lambda u,t: cp.asarray(f(u,t), float)
        self.f = f
        self.time_step = time_step # size of time step
        self.t = 0

    def advance(self):
        k = self.k
        dt = self.time_step
        unew = self.u[k] + dt*self.f(self.u[k],self.t[k])
        return unew

class Interface(object):
    def __init__(self, internal_variable, external_variable):
        self.external_variable = external_variable
        self.internal_variable = internal_variable
        if self.internal_variable is None:

            if len(external_variable.shape) == 2:
                self.internal_variable = cp.zeros(tuple(self.external_variable.shape))
                self.internal_variable = self.internal_variable[:,:,cp.newaxis]
                self.external_variable_index = 0
            else:
                self.internal_variable = cp.zeros(tuple(self.external_variable.shape))
                self.external_variable_index = slice(0,self.external_variable.shape[2],1)

        else:

            new_internal_variable = cp.concatenate((self.internal_variable, external_variable), axis = 2)
            self.external_variable_index = slice(self.internal_variable.shape[2], new_internal_variable.shape[2],1)
            self.internal_variable = new_internal_variable


    def update_internal_variable(self):
        #print("internal: ", self.internal_variable.shape)
        #print("external: ", self.external_variable.shape)

        self.internal_variable[:,:,self.external_variable_index] = self.external_variable

def cap_array(array, upper_cap):
        below_upper_limit = array < upper_cap
        array *= below_upper_limit
        array += (below_upper_limit == 0)*upper_limit

class Integrate_and_fire_neuron_membrane_function(object):
    def __init__(self, leakage_reversal_potential, membrane_resistance, membrane_time_constant, summed_inputs):
        self.leakage_reversal_potential = leakage_reversal_potential    # E_m
        self.membrane_resistance = membrane_resistance                  # R_m
        self.membrane_time_constant = membrane_time_constant            # tau_m
        self.summed_inputs = summed_inputs

    def __call__(self,V,t):

        delta_V = (self.leakage_reversal_potential - V + self.membrane_resistance * self.summed_inputs)/self.membrane_time_constant

        return delta_V

class Circuit_Equation(object):
    def __init__(self, input_resistance, time_constant, summed_inputs):

        self.input_resistance = input_resistance
        self.time_constant = time_constant
        self.summed_inputs = summed_inputs

    def __call__(self, V,t):

        delta_V = (self.input_resistance * self.summed_inputs - V) / self.time_constant

        return delta_V


'''
Components
'''
class Component(object):
    def interface(self, input_object, delayed_input):
        pass
class Simple_Integrate_and_fire_soma(object):
    def __init__(self, parameter_dict):
        self.parameters = parameter_dict

        if len(self.parameters["population_size"]) == 2:
            self.population_size = self.parameters["population_size"]
        else:
            print("Population size must be size 2 and give population size in x and y dimensions")
            sys.exit(0)

        self.time_step = self.parameters["time_step"]
        self.refractory_period = self.parameters["refractory_period"]
        self.time_since_last_spike = cp.ones(self.population_size) + self.refractory_period + 1
        self.current_somatic_voltages = cp.zeros(self.parameters["population_size"])
        self.new_somatic_voltages = cp.zeros(self.parameters["population_size"])

        self.new_spiked_neurons = cp.zeros(self.parameters["population_size"])
        self.current_spiked_neurons = cp.zeros(self.parameters["population_size"])

        self.input_interfaces = []
        self.summed_inputs = cp.zeros((self.parameters["population_size"]))
        self.inputs = None

        membrane_function = Circuit_Equation(self.parameters["input_resistance"], self.parameters["membrane_time_constant"], self.summed_inputs)
        self.membrane_solver = RungeKutta2_cupy(membrane_function, self.parameters["time_step"])
        self.threshold = self.parameters["threshold"]
        self.reset_voltage = self.parameters["reset_voltage"]

        self.membrane_solver.set_initial_condition(self.current_somatic_voltages)

        self.dead_cells_location = 1

    def interface_input(self, external_variable):
        self.input_interfaces.append(Interface(self.inputs, external_variable))

        # update inputs for everything that uses it
        self.inputs = self.input_interfaces[-1].internal_variable
        for interface_input in self.input_interfaces:
            interface_input.internal_variable = self.inputs

    def set_dead_cells(self,dead_cells_location):
        self.dead_cells_location = dead_cells_location == 0

    def compute_new_values(self):

        self.time_since_last_spike += self.time_step
        self.new_somatic_voltages += self.membrane_solver.advance(self.current_somatic_voltages, t = 0)

        # set somatic values for neurons that have fired within the refractory period to zero
        self.new_somatic_voltages *= self.time_since_last_spike > self.refractory_period

        self.new_spiked_neurons[:,:] = self.new_somatic_voltages > self.threshold

        non_spike_mask = self.new_spiked_neurons == 0
        self.time_since_last_spike *= non_spike_mask

        self.new_somatic_voltages *= non_spike_mask
        self.new_somatic_voltages += self.new_spiked_neurons * self.reset_voltage

        # destroy values in dead cells

        self.new_somatic_voltages *= self.dead_cells_location
        self.new_spiked_neurons *= self.dead_cells_location

        # set this to avoid overlflow
        below_upper_limit = self.time_since_last_spike < upper_limit
        self.time_since_last_spike *= below_upper_limit
        self.time_since_last_spike += (below_upper_limit == 0)*upper_limit

    def update_current_values(self):

        for interface_variable in self.input_interfaces:
            interface_variable.update_internal_variable()


        if len(self.inputs.shape) == 3:
            self.summed_inputs[:,:] = cp.sum(self.inputs, axis = 2)
        else:
            self.summed_inputs[:,:] = self.inputs

        self.current_somatic_voltages[:,:] = self.new_somatic_voltages
        self.current_spiked_neurons[:,:] = self.new_spiked_neurons

class Dynamical_Axonal_Terminal_Markram_etal_1998(object):
    def __init__(self, parameter_dict):
        self.parameters = parameter_dict

        self.spike_matrix = None
        self.delta_t = self.parameters["time_step"]



    def interface_read_variable(self, read_variable):
        # read variable should be a 2d or 3d array containing boolean values of spikes
        self.spike_matrix = read_variable

        self.population_size = self.spike_matrix.shape
        self.time_since_last_spike = cp.zeros(self.population_size)


        if self.parameters["resting_utilization_of_synaptic_efficacy"]["distribution"] == "normal":
            self.resting_utilization_of_synaptic_efficacy = cp.random.normal(self.parameters["resting_utilization_of_synaptic_efficacy"]["mean"], self.parameters["resting_utilization_of_synaptic_efficacy"]["SD"], self.population_size)

            negative_values = self.resting_utilization_of_synaptic_efficacy <= 0
            replacement_values = cp.random.uniform(self.parameters["resting_utilization_of_synaptic_efficacy"]["mean"] - self.parameters["resting_utilization_of_synaptic_efficacy"]["SD"], self.parameters["resting_utilization_of_synaptic_efficacy"]["mean"] + self.parameters["resting_utilization_of_synaptic_efficacy"]["SD"], self.population_size)
            self.resting_utilization_of_synaptic_efficacy *= negative_values == 0
            self.resting_utilization_of_synaptic_efficacy += replacement_values*negative_values

        else:
            print("only normal distribution implementee for resting_utilization_of_synaptic_efficacy_distribution")
            sys.exit(0)

        if self.parameters["absolute_synaptic_efficacy"]["distribution"] == "normal":
            self.weight_matrix = cp.random.normal(self.parameters["absolute_synaptic_efficacy"]["mean"], self.parameters["absolute_synaptic_efficacy"]["SD"], self.population_size)

            if self.parameters["type"] == "excitatory":
                negative_values = self.weight_matrix <= 0
                replacement_values = cp.random.uniform(self.parameters["absolute_synaptic_efficacy"]["mean"] - self.parameters["absolute_synaptic_efficacy"]["SD"], self.parameters["absolute_synaptic_efficacy"]["mean"] + self.parameters["absolute_synaptic_efficacy"]["SD"], self.population_size)
                self.weight_matrix *= negative_values == 0
                self.weight_matrix += replacement_values*negative_values
            elif self.parameters["type"] == "inhibitory":
                positive_values = self.weight_matrix <= 0
                replacement_values = cp.random.uniform(self.parameters["absolute_synaptic_efficacy"]["mean"] - self.parameters["absolute_synaptic_efficacy"]["SD"], self.parameters["absolute_synaptic_efficacy"]["mean"] + self.parameters["absolute_synaptic_efficacy"]["SD"], self.population_size)
                self.weight_matrix *= positive_values == 0
                self.weight_matrix += replacement_values*positive_values

        else:
            print("Absolute synaptic efficacy distributions other than normal has not been implemented")
            sys.exit(0)

        if self.parameters["time_constant_depresssion"]["distribution"] == "normal":
            self.tau_recovery = cp.random.normal(self.parameters["time_constant_depresssion"]["mean"], self.parameters["time_constant_depresssion"]["SD"], self.population_size)

            negative_values = self.tau_recovery <= 0
            replacement_values = cp.random.uniform(self.parameters["time_constant_depresssion"]["mean"] - self.parameters["time_constant_depresssion"]["SD"], self.parameters["time_constant_depresssion"]["mean"] + self.parameters["time_constant_depresssion"]["SD"], self.population_size)
            self.tau_recovery *= negative_values == 0
            self.tau_recovery += replacement_values*negative_values

        else:
            print("Only normal distribution implemented for time_constant_depression")

        if self.parameters["time_constant_facilitation"]["distribution"] == "normal":
            self.tau_facil = cp.random.normal(self.parameters["time_constant_facilitation"]["mean"], self.parameters["time_constant_facilitation"]["SD"], self.population_size )

            negative_values = self.tau_facil <= 0
            replacement_values = cp.random.uniform(self.parameters["time_constant_facilitation"]["mean"] - self.parameters["time_constant_facilitation"]["SD"], self.parameters["time_constant_facilitation"]["mean"] + self.parameters["time_constant_facilitation"]["SD"], self.population_size)
            self.tau_facil *= negative_values == 0
            self.tau_facil += replacement_values*negative_values

        else:
            print("Only normal distribution implemented for time_constant_depression")

        self.current_neurotransmitter_reserve = cp.ones(self.population_size) # R
        self.new_neurotransmitter_reserve = cp.ones(self.population_size)

        self.current_utilization_of_synaptic_efficacy = cp.ones(self.population_size) + self.resting_utilization_of_synaptic_efficacy
        self.new_utilization_of_synaptic_efficacy = cp.ones(self.population_size)

        self.current_synaptic_response = cp.zeros(self.population_size)
        self.new_synaptic_response = cp.zeros(self.population_size)

        if cp.any(self.tau_recovery <= 0) or cp.any(self.tau_facil <= 0) or cp.any(self.resting_utilization_of_synaptic_efficacy <= 0):
            print("unsuccefull at removing negative values")
            sys.exit(0)

    def compute_new_values(self):


        self.new_utilization_of_synaptic_efficacy[:,:,:] = self.current_utilization_of_synaptic_efficacy * cp.exp((-self.time_since_last_spike) / self.tau_facil) + self.resting_utilization_of_synaptic_efficacy*(1 - self.current_utilization_of_synaptic_efficacy * cp.exp((-self.time_since_last_spike) / self.tau_facil))



        self.new_neurotransmitter_reserve[:,:,:] = self.current_neurotransmitter_reserve * (1 - self.new_utilization_of_synaptic_efficacy)*cp.exp(-self.time_since_last_spike / self.tau_recovery) + 1 - cp.exp(-self.time_since_last_spike / self.tau_recovery)


        self.time_since_last_spike += self.delta_t
        self.time_since_last_spike *= self.spike_matrix == 0

        self.new_synaptic_response[:,:,:] = self.weight_matrix * self.new_utilization_of_synaptic_efficacy *self.new_neurotransmitter_reserve*self.spike_matrix

        no_spike_mask = self.spike_matrix == 0
        self.new_utilization_of_synaptic_efficacy *= self.spike_matrix
        self.new_utilization_of_synaptic_efficacy += self.current_utilization_of_synaptic_efficacy*no_spike_mask

        self.new_neurotransmitter_reserve *= self.spike_matrix
        self.new_neurotransmitter_reserve += self.current_neurotransmitter_reserve*no_spike_mask


    def update_current_values(self):
        self.current_synaptic_response[:,:,:] = self.new_synaptic_response

        self.current_utilization_of_synaptic_efficacy[:,:,:] = self.new_utilization_of_synaptic_efficacy

        self.current_neurotransmitter_reserve[:,:,:] = self.new_neurotransmitter_reserve


#####

class Dendritic_Spine_Maas(object):
    def __init__(self, parameter_dict):
        self.parameters = parameter_dict
        self.dt = self.parameters["time_step"]
        self.time_constant = self.parameters["time_constant"]

        pass
    def interface_read_variable(self, read_variable):
        self.current_synaptic_input = read_variable

        self.population_size = read_variable.shape
        self.last_input_since_spike = cp.zeros(self.population_size)
        self.new_synaptic_output = cp.zeros(self.population_size)
        self.current_synaptic_output = cp.zeros(self.population_size)

        self.time_since_last_spike = cp.ones(self.population_size) + 1000

        pass
    def compute_new_values(self):
        # compute new time since last spiked first to decay current value
        self.time_since_last_spike += self.dt

        self.new_synaptic_output[:,:,:] = self.last_input_since_spike * cp.exp(-self.time_since_last_spike / self.time_constant)
        self.new_synaptic_output += self.current_synaptic_input

        current_input_mask = self.current_synaptic_input == 0
        self.last_input_since_spike *= current_input_mask
        self.last_input_since_spike += self.new_synaptic_output * (current_input_mask == 0)
        self.time_since_last_spike *= current_input_mask
        cap_array(self.time_since_last_spike,10000)



    def update_current_values(self):
        self.current_synaptic_output[:,:,:] = self.new_synaptic_output




class Dendritic_Arbor(object):
    def __init__(self, parameter_dict):
        self.parameters = parameter_dict
        self.projection_template = self.parameters["projection_template"]

    def interface(self, read_variable):
        # read variable should be a 2d array containing spikes
        self.axonal_hillock_spikes_array = read_variable


        print("Arborizing axon \n")
        if len(self.projection_template.shape) <= 1:
            print("Projection template has 1 axis")
            self.template_rolls = [[0,0]]
            self.midX = 0
            self.midY = 0
            self.max_level = 1
            if len(self.axonal_hillock_spikes_array.shape) <= 1:
                print("axonal_hillock_spikes_array has 1 axis of size: ", self.axonal_hillock_spikes_array.shape)
                self.new_spike_array = cp.zeros(self.axonal_hillock_spikes_array.shape[0],dtype='float64')
                self.current_spike_array = cp.zeros(self.axonal_hillock_spikes_array.shape[0],dtype='float64')
            else:
                print("axonal_hillock_spikes_array has 2 axis of size: ", self.inputs.shape)
                self.new_spike_array = cp.zeros((self.axonal_hillock_spikes_array.shape[0], self.axonal_hillock_spikes_array.shape[1]) )
                self.current_spike_array = cp.zeros((self.axonal_hillock_spikes_array.shape[0], self.axonal_hillock_spikes_array.shape[1]) )
        else:
            print("Neihbourhood_template has 2 axis: \n ###################### \n", self.projection_template)
            print("######################")
            self.midX = int(-self.projection_template.shape[0]/2)
            self.midY = int(-self.projection_template.shape[1]/2)

            self.template_rolls = []
            self.max_level = 0
            for i0 in range(self.projection_template.shape[0]):
                for i1 in range(self.projection_template.shape[1]):
                    if self.projection_template[i0,i1] == 1:
                        self.template_rolls.append([self.midX + i0, self.midY + i1])
                        self.max_level += 1

            if len(self.axonal_hillock_spikes_array.shape) <= 1:
                print("axonal_hillock_spikes_array have 1 axis of length: ", self.axonal_hillock_spikes_array.shape)
                self.new_spike_array = cp.zeros((self.axonal_hillock_spikes_array.shape[0], self.max_level))
                self.current_spike_array = cp.zeros((self.axonal_hillock_spikes_array.shape[0], self.max_level))
            elif (len(self.axonal_hillock_spikes_array.shape) == 2):
                print("axonal_hillock_spikes_array have 2 axis of shape: ", self.axonal_hillock_spikes_array.shape)
                self.new_spike_array = cp.zeros((self.axonal_hillock_spikes_array.shape[0], self.axonal_hillock_spikes_array.shape[1], self.max_level))
                self.current_spike_array = cp.zeros((self.axonal_hillock_spikes_array.shape[0], self.axonal_hillock_spikes_array.shape[1], self.max_level))
            else:
                print("######################### \n Error! \n #############################")
                print("axonal_hillock_spikes_array have more than 2 axis: ", self.axonal_hillock_spikes_array.shape)
            # compute a list that gives the directions a spike should be sent to
            self.population_size = self.current_spike_array.shape

            self.template_rolls = cp.array(self.template_rolls)
            self.kill_mask = cp.ones(self.population_size)

    def kill_connections_based_on_distance(self):
        C = self.parameters["distance_based_connection_probability"]["C"]
        lambda_parameter = self.parameters["distance_based_connection_probability"]["lambda_parameter"]

        distance = cp.linalg.norm(self.template_rolls, ord = 2, axis = 1)


        self.kill_mask = cp.random.uniform(0,1,self.population_size)

        for distance_index in range(self.population_size[2]):
            self.kill_mask[:,:,distance_index] = self.kill_mask[:,:,distance_index] < C* cp.exp(-(distance[distance_index]/lambda_parameter)**2)

    def compute_new_values(self):
        #print(self.axonal_hillock_spikes_array.shape)
        if self.max_level <= 1:
            self.new_spike_array[:,:] = self.axonal_hillock_spikes_array
        else:
            for i0, x_y in enumerate(self.template_rolls):
                #To do: probably a bad solution to do this in two operations, should try to do it in one

                axonal_hillock_spikes_array_rolled = cp.roll(self.axonal_hillock_spikes_array, int(x_y[0]), axis = 0)
                axonal_hillock_spikes_array_rolled = cp.roll(axonal_hillock_spikes_array_rolled, int(x_y[1]), axis = 1)

                self.new_spike_array[:,:,i0] = axonal_hillock_spikes_array_rolled
        self.new_spike_array *= self.kill_mask

    def update_current_values(self):
        if self.max_level <= 1:
            self.current_spike_array[:,:] = self.axonal_hillock_spikes_array
        else:
            self.current_spike_array[:,:,:] = self.new_spike_array

class Delay_Line(object):
    def __init__(self, parameter_dict):
        self.parameters = parameter_dict

        self.delay_in_compute_steps = int(self.parameters["delay"] / self.parameters["time_step"])

    def interface_read_variable(self, read_variable):
        # read_variable should be a 2d array of spikes
        self.spike_source = read_variable

        self.delay_line = cp.zeros((self.spike_source.shape[0], self.spike_source.shape[1], self.delay_in_compute_steps))
        self.new_spike_output = cp.zeros(self.spike_source.shape)
        self.current_spike_output = cp.zeros(self.spike_source.shape)


    def compute_new_values(self):
        self.delay_line[:,:,:] = cp.roll(self.delay_line,1, axis = 2)
        self.new_spike_output[:,:] = self.delay_line[:,:,-1]
        self.delay_line[:,:,0] = self.spike_source

    def update_current_values(self):
        self.current_spike_output[:,:] = self.new_spike_output

class Neuron(object):
    def __init__(self):
        self.component_list = []

    def compute_new_values(self):
        for component in self.component_list:
            component.compute_new_values()
    def update_current_values(self):
        for component in self.component_list:
            component.update_current_values()


class Readout_P_Delta(object):
    def __init__(self, parameter_dict):
        self.parameters = parameter_dict

        self.nr_of_readout_neurons = self.parameters["nr_of_readout_neurons"]
        self.parallel_perceptron_outputs = cp.zeros(self.nr_of_readout_neurons)

        self.squashing_function = Squashing_Function_rho(self.parameters["rho"])
        self.margin = self.parameters["margin"]
        # gamma in paper

        self.clear_margin_importance = self.parameters["clear_margins_importance"]
        # mu in paper

        self.error_tolerance = self.parameters["error_tolerance"]
        # small epsilon in paper

        self.learning_rate = self.parameters["learning_rate"]
        # eta in paper

    def activation_function(self):
        input_projection = cp.repeat(self.inputs[:,:,cp.newaxis], self.nr_of_readout_neurons, axis = 2)
        parallel_perceptron_outputs = cp.sum(input_projection*self.weights, axis = (0,1))
        return parallel_perceptron_outputs

    def update_weights(self, desired_output):
        self.desired_output = desired_output

        #testing fic
        input_projection = cp.repeat(self.inputs[:,:,cp.newaxis], self.nr_of_readout_neurons, axis = 2)
        parallel_perceptron_outputs = cp.sum(input_projection*self.weights, axis = (0,1))
        #self.parallel_perceptron_outputs *= 0.3
        #self.parallel_perceptron_outputs += parallel_perceptron_outputs

        #parallel_perceptron_outputs = self.parallel_perceptron_outputs

        #parallel_perceptron_outputs = self.activation_function()

        # summary rule 1
        parallel_perceptron_output_above_equal_0 = parallel_perceptron_outputs >= 0
        # adding axis and transposing to allow the array to be multiplied with 3d input array correctly
        parallel_perceptron_output_above_equal_0 = parallel_perceptron_output_above_equal_0[cp.newaxis, cp.newaxis,:].T.T

        #summary rule 2
        parallel_perceptron_output_below_0 = parallel_perceptron_outputs < 0
        parallel_perceptron_output_below_0 = parallel_perceptron_output_below_0[cp.newaxis, cp.newaxis,:].T.T
        # summary rule 3, note: margin is yotta in paper
        parallel_perceptron_output_above_0_below_margin = parallel_perceptron_outputs >= 0
        parallel_perceptron_output_above_0_below_margin *= parallel_perceptron_output_above_0_below_margin < self.margin
        parallel_perceptron_output_above_0_below_margin = parallel_perceptron_output_above_0_below_margin[cp.newaxis, cp.newaxis,:].T.T

        # summary rule 4
        parallel_perceptron_output_below_0_above_neg_margin = parallel_perceptron_outputs < 0
        parallel_perceptron_output_below_0_above_neg_margin *= parallel_perceptron_outputs > -1*self.margin
        parallel_perceptron_output_below_0_above_neg_margin = parallel_perceptron_output_below_0_above_neg_margin[cp.newaxis, cp.newaxis,:].T.T

        # summary rule 5
        #zeros
        weight_update_direction = cp.zeros(self.weight_shape)


        population_output = cp.sum(parallel_perceptron_output_above_equal_0) - cp.sum(parallel_perceptron_output_below_0)
        population_output = self.squashing_function(population_output)

        # compute the lower limits first and then the higher


        if population_output > self.desired_output + self.error_tolerance:

            weight_update_direction += (-1) * input_projection * parallel_perceptron_output_above_equal_0

        elif population_output < self.desired_output - self.error_tolerance:

            masked_input_projection = input_projection * parallel_perceptron_output_below_0
            weight_update_direction += masked_input_projection

        if population_output >= (self.desired_output - self.error_tolerance):
            weight_update_direction += self.clear_margin_importance*(-1 * input_projection) * parallel_perceptron_output_below_0_above_neg_margin

        if population_output <= self.desired_output + self.margin:

            weight_update_direction += self.clear_margin_importance * input_projection * parallel_perceptron_output_above_0_below_margin



        # something strange is happening with the weight update. Testing with random update to see if it is an issue with the accuracy calculation
        #weight_update_direction = cp.random.uniform(-1,1,self.weight_shape)
        weight_update_direction *= self.learning_rate


        weight_bounding = self.weights.reshape(self.weights.shape[0]*self.weights.shape[1], self.weights.shape[2])
        weight_bounding = (cp.linalg.norm(weight_bounding, ord = 2, axis = 0)**2 - 1)
        #print(weight_bounding)
        weight_bounding = weight_bounding[cp.newaxis, cp.newaxis,:].T.T
        weight_bounding *= self.learning_rate
        weight_bounding = self.weights * weight_bounding

        # update weights
        self.weights -= weight_bounding
        self.weights += weight_update_direction

        self.current_population_output = population_output

    def classify(self, image):
        input_projection = cp.repeat(image[:,:,cp.newaxis], self.nr_of_readout_neurons, axis = 2)
        parallel_perceptron_outputs = cp.sum(input_projection*self.weights, axis = (0,1))

        parallel_perceptron_output_above_equal_0 = parallel_perceptron_outputs >= 0
        parallel_perceptron_output_below_0 = parallel_perceptron_outputs < 0


        population_output = cp.sum(parallel_perceptron_output_above_equal_0) - cp.sum(parallel_perceptron_output_below_0)
        population_output = self.squashing_function(population_output)

        return population_output

    def interface_read_variable(self, read_variable):
        # read_variable is a 2d array of spikes
        self.inputs = read_variable

        self.weight_shape = list(self.inputs.shape)
        #print("list weight shape ", self.weight_shape)
        self.weight_shape.append(self.nr_of_readout_neurons)
        #print("appended list weight shape ", self.weight_shape)
        self.weights = cp.random.uniform(-1,1,self.weight_shape)

class P_Delta_I_n_F_Neurons(Readout_P_Delta):
    def __init__(self, parameter_dict):
        super.__init__(parameter_dict)

        membrane_function = Circuit_Equation(self.parameters["input_resistance"], self.parameters["membrane_time_constant"], self.summed_inputs)
        self.membrane_solver = RungeKutta2_cupy(membrane_function, self.parameters["time_step"])

    def activation_function(self, input_projection):

        parallel_perceptron_outputs = cp.sum(input_projection*self.weights, axis = (0,1))

        self.somatic_voltages += self.membrane_solver.advance(self.current_somatic_voltages, t = 0)

    def interface_read_variable(self, read_variable):
        super.interface_read_variable(read_variable)
        self.somatic_voltages = cp.zeros(read_variable.shape)

class Squashing_Function_rho(object):
    def __init__(self, rho):
        self.rho = rho
    def __call__(self, input_values):
        below_neg_rho = input_values < (-1* self.rho)
        above_pos_rho = input_values > self.rho

        between_rho = (below_neg_rho + above_pos_rho) == 0

        output = (below_neg_rho*-1) + above_pos_rho + (input_values * between_rho)/self.rho
        return output


### Experimental stuff

class Readout_P_Delta_prototype_learner(object):
    def __init__(self, parameter_dict):
        self.parameters = parameter_dict

        self.nr_of_readout_neurons = self.parameters["nr_of_readout_neurons"]

        self.squashing_function = Squashing_Function_rho(self.parameters["rho"])
        self.margin = self.parameters["margin"]
        # gamma in paper

        self.clear_margin_importance = self.parameters["clear_margins_importance"]
        # mu in paper

        self.error_tolerance = self.parameters["error_tolerance"]
        # small epsilon in paper

        self.learning_rate = self.parameters["learning_rate"]
        # eta in paper

    def update_weights(self, desired_output):
        self.desired_output = desired_output

        input_projection = cp.repeat(self.inputs[:,:,cp.newaxis], self.nr_of_readout_neurons, axis = 2)
        parallel_perceptron_outputs = 1/ (cp.sum(input_projection - self.weights, axis = (0,1))**2 +1)

        # summary rule 1
        parallel_perceptron_output_above_equal_0 = parallel_perceptron_outputs >= 0
        # adding axis and transposing to allow the array to be multiplied with 3d input array correctly
        parallel_perceptron_output_above_equal_0 = parallel_perceptron_output_above_equal_0[cp.newaxis, cp.newaxis,:].T.T

        #summary rule 2
        parallel_perceptron_output_below_0 = parallel_perceptron_outputs < 0
        parallel_perceptron_output_below_0 = parallel_perceptron_output_below_0[cp.newaxis, cp.newaxis,:].T.T
        # summary rule 3, note: margin is yotta in paper
        parallel_perceptron_output_above_0_below_margin = parallel_perceptron_outputs >= 0
        parallel_perceptron_output_above_0_below_margin *= parallel_perceptron_output_above_0_below_margin < self.margin
        parallel_perceptron_output_above_0_below_margin = parallel_perceptron_output_above_0_below_margin[cp.newaxis, cp.newaxis,:].T.T

        # summary rule 4
        parallel_perceptron_output_below_0_above_neg_margin = parallel_perceptron_outputs < 0
        parallel_perceptron_output_below_0_above_neg_margin *= parallel_perceptron_outputs > -1*self.margin
        parallel_perceptron_output_below_0_above_neg_margin = parallel_perceptron_output_below_0_above_neg_margin[cp.newaxis, cp.newaxis,:].T.T

        # summary rule 5
        #zeros
        weight_update_direction = cp.zeros(self.weight_shape)


        population_output = cp.sum(parallel_perceptron_output_above_equal_0) - cp.sum(parallel_perceptron_output_below_0)
        population_output = self.squashing_function(population_output)

        # compute the lower limits first and then the higher
        if population_output >= (self.desired_output - self.error_tolerance):

            weight_update_direction += self.clear_margin_importance*(-1 * input_projection) * parallel_perceptron_output_below_0_above_neg_margin

            if population_output <= self.desired_output + self.margin:

                weight_update_direction += self.clear_margin_importance * input_projection * parallel_perceptron_output_above_0_below_margin

            elif population_output > self.desired_output + self.error_tolerance:

                weight_update_direction += (-1) * input_projection * parallel_perceptron_output_above_equal_0

        elif population_output < self.desired_output - self.error_tolerance:

            masked_input_projection = input_projection * parallel_perceptron_output_below_0
            weight_update_direction += masked_input_projection


        weight_update_direction *= self.learning_rate
        self.weights += weight_update_direction

        weight_bounding = self.weights.reshape(self.weights.shape[0]*self.weights.shape[1], self.weights.shape[2])
        weight_bounding = (cp.linalg.norm(weight_bounding, ord = 2, axis = 0)**2 - 1)
        #print(weight_bounding)
        weight_bounding = weight_bounding[cp.newaxis, cp.newaxis,:].T.T
        weight_bounding *= self.learning_rate
        weight_bounding = self.weights * weight_bounding


        self.weights -= weight_bounding

        self.current_population_output = population_output

    def classify(self, image):
        input_projection = cp.repeat(self.inputs[:,:,cp.newaxis], self.nr_of_readout_neurons, axis = 2)
        parallel_perceptron_outputs = cp.sum(input_projection*self.weights, axis = (0,1))
        population_output = cp.sum(parallel_perceptron_outputs)
        population_output = self.squashing_function(population_output)
        return population_output
    def interface_read_variable(self, read_variable):
        # read_variable is a 2d array of spikes
        self.inputs = read_variable

        self.weight_shape = list(self.inputs.shape)
        print("list weight shape ", self.weight_shape)
        self.weight_shape.append(self.nr_of_readout_neurons)
        print("appended list weight shape ", self.weight_shape)
        self.weights = cp.random.uniform(-1,1,self.weight_shape)



'''

population_size = (500,500)
threshold = 20 # mV
reset_voltage = 0 #mV
background_current = 13.5 #nA
input_resistance = 1 #MOhm
leakage_reversal_potential = 0
membrane_resistance = 10
membrane_time_constant =10 #ms
time_step = 0.1 #ms

neurons = Simple_Integrate_and_fire_soma(population_size, time_step, reset_voltage, threshold, leakage_reversal_potential, membrane_resistance, membrane_time_constant)

inputs = cp.zeros(population_size)

neurons.interface_input(inputs)

while True:
    inputs[:,:] = 2.02# cp.random.rand(population_size[0], population_size[1])
    neurons.compute_new_values()
    neurons.update_current_values()
    neurons.current_somatic_voltages[250,250] = threshold
    cv2.imshow('frame', neurons.current_somatic_voltages)
    print(neurons.current_somatic_voltages[0,0], inputs[0,0], neurons.summed_inputs[0,0])
    #network.network_state[0,:] = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
'''
