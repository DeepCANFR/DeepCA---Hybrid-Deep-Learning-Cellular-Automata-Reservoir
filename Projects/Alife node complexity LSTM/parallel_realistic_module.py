import numpy as np
import cupy as cp
import cupy as ncp
# Note: ncp is supposed to offer the option of using either numpy or cupy as dropin for computations
import dask
import string
import random
import time

'''
Differential equiation solvers
'''
class RungeKutta2_cupy(object):
    def __init__(self, f, time_step):
        # Initialize the class with the size of the time steps you are using

        self.f = f
        self.time_step = time_step # size of time step
        self.t = 0 # starting time

    def advance(self, u, t):

        K1 = self.time_step * (self.f(u,t))
        K2 = self.time_step * self.f(u + (1/2)* K1, t + (1/2)*self.time_step)

        t += self.time_step
        u_delta = K2

        return u_delta


class ForwardEuler_cupy(object):
    def __init__(self, f, time_step):
        self.f = f
        self.time_step = time_step # size of time step
        self.t = 0

    def advance(self):
        k = self.k
        dt = self.time_step
        unew = self.u[k] + dt*self.f(self.u[k],self.t[k])
        return unew


'''
Membrane functions
'''
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

class Izhivechik_Equation(object):
    def __init__(self, a, b, summed_inputs, population_size):
        self.a = a
        self.b = b

        self.summed_inputs = summed_inputs
        self.population_size = population_size

    def __call__(self, v_u, t = 0):

        delta_v_u = ncp.zeros((self.population_size[0], self.population_size[1], 2))
        delta_v_u[:,:,0] = 0.04*v_u[:,:,0]**2 + 5*v_u[:,:,0] + 140 - v_u[:,:,1] + self.summed_inputs
        delta_v_u[:,:,1] = self.a * (self.b * v_u[:,:,0] - v_u[:,:,1])
        return delta_v_u

'''
Spike generators
'''
class Poisson_Spike_Generator(object):
    def __init__(self, scale, size, refractory_period):
        self.refractory_period = refractory_period
        self.scale = scale
        self.size = size
        self.last_spike_time = ncp.zeros(size, dtype = ncp.float64)
        self.next_spike_time = self.refractory_period + ncp.random.exponential(self.scale, self.size)
    def homogenous_poisson_spike(self, t):
        new_spikes = self.next_spike_time < t
        new_spikes_mask = new_spikes == 0
        self.last_spike_time *= new_spikes_mask*1.0
        self.last_spike_time += new_spikes*t
        new_next_spike_time = self.refractory_period + ncp.random.exponential(self.scale, self.size)
        self.next_spike_time += new_spikes * new_next_spike_time
        print(t, self.next_spike_time, new_spikes)
        return new_spikes





'''
Support classes
'''
class Interfacable_Array(object):
    def __init__(self, population_shape):
        self.array = ncp.zeros(population_shape)
        self.array = self.array[:,:,ncp.newaxis]


        self.external_components = []
        self.external_components_indexes = []


    def interface(self, external_component):

        external_interface = external_component.interfacable
        external_interface_shape = external_interface.shape

        if len(self.external_components) == 0 :
            if len(external_interface_shape) == 2:
                self.array = ncp.zeros(external_interface_shape)
                self.array = self.array[:,:,ncp.newaxis]
                self.external_components_indexes.append(slice(0,1,1))

            else:
                self.array = ncp.zeros(external_interface_shape)
                self.external_components_indexes.append(slice(0,external_interface_shape[2],1))

        else:
            if len(external_interface_shape) == 2:
                old_axis_2_length = self.array.shape[2]
                self.array = ncp.concatenate((self.array, external_interface[:,:,ncp.newaxis]), axis = 2)
                self.external_components_indexes.append(slice(old_axis_2_length, self.array.shape[2],1))

            else:
                old_axis_2_length = self.array.shape[2]
                self.array = ncp.concatenate((self.array, external_interface), axis = 2)
                self.external_components_indexes.append(slice(old_axis_2_length, self.array.shape[2],1))

        self.external_components.append(external_component)


    def update(self):

        for index, external_component in enumerate(self.external_components):
            component_index = self.external_components_indexes[index]
            external_interface = external_component.interfacable

            if len(external_interface.shape) == 2:
                external_interface = external_interface[:,:, ncp.newaxis]
                self.array[:,:,component_index] = external_interface
            else:
                self.array[:,:,component_index] = external_interface


    def get_sum(self):
        return ncp.sum(self.array, axis = 2)


'''
Component classes
##################################################################
'''

class Component(object):
    interfacable = 0
    component_IDs = []
    '''
    This is the base class for all components
    Every component must implement the functions given below
    '''

    def __init__(self, parameter_dict):
        self.parameters = parameter_dict
        self.realized_parameters = {}
        time.sleep(0.5)



    def interface(self, component):
        raise NotImplementedError

    def compute_new_values(self):
        raise NotImplementedError
    def update_current_values(self):
        raise NotImplementedError

    def create_indexes(self, shape):
        indexes = []
        for _ in range(len(shape)):
            indexes.append(slice(0,None,1))
        return tuple(indexes)

'''
Somas
'''
class Base_Integrate_and_Fire_Soma(Component):
    interfacable = 0
    current_somatic_voltages = 0
    current_spiked_neurons = 0
    new_spiked_neurons = 0
    new_somatic_voltages = 0
    current_u = 0
    summed_inputs = 0
    def __init__(self, parameter_dict):
        super().__init__(parameter_dict)

        if len(self.parameters["population_size"]) == 2:
            self.population_size = self.parameters["population_size"]
        else:
            print("Population size must be size 2 and give population size in x and y dimensions")
            sys.exit(0)

        self.time_step = self.parameters["time_step"]
        self.refractory_period = self.parameters["refractory_period"]
        self.time_since_last_spike = ncp.ones(self.population_size) + self.refractory_period + 1
        self.current_somatic_voltages = ncp.ones(self.parameters["population_size"])*self.parameters["reset_voltage"]
        self.new_somatic_voltages = ncp.ones(self.parameters["population_size"])*self.parameters["reset_voltage"]

        self.new_spiked_neurons = ncp.zeros(self.parameters["population_size"])
        self.current_spiked_neurons = ncp.zeros(self.parameters["population_size"])

        ## needs fixing
        self.inputs = Interfacable_Array(self.population_size)

        self.summed_inputs = ncp.zeros((self.parameters["population_size"]))
        self.threshold = self.parameters["threshold"]
        self.reset_voltage = self.parameters["reset_voltage"]

        #self.membrane_solver.set_initial_condition(self.current_somatic_voltages)

        self.dead_cells_location = 1
        self.upper_limit = self.parameters["temporal_upper_limit"]
        self.interfacable = self.current_spiked_neurons
        self.set_membrane_function()

    def set_membrane_function(self):
        raise NotImplementedError
        sys.exit(1)

    def interface(self, external_component):
        self.inputs.interface(external_component)

    def set_dead_cells(self,dead_cells_location):
        self.dead_cells_location = dead_cells_location == 0

    def cap_array(self, array, upper_limit):
        below_upper_limit = array < upper_limit
        array *= below_upper_limit
        array += (below_upper_limit == 0)*upper_limit
        return array

    def reset_spiked_neurons(self):
        non_spike_mask = self.new_spiked_neurons == 0
        self.time_since_last_spike *= non_spike_mask

        self.new_somatic_voltages *= non_spike_mask
        self.new_somatic_voltages += self.new_spiked_neurons * self.reset_voltage

    def kill_dead_values(self):
        # destroy values in dead cells
        self.new_somatic_voltages *= self.dead_cells_location
        self.new_spiked_neurons *= self.dead_cells_location
    def set_refractory_values(self):
        # set somatic voltages to the reset value if within refractory period
        self.new_somatic_voltages *= self.time_since_last_spike > self.parameters["refractory_period"]
        self.new_somatic_voltages += (self.time_since_last_spike <= self.parameters["refractory_period"]) * self.parameters["reset_voltage"]

    def compute_new_values(self):
        raise NotImplementedError
        sys.exit(1)


    def update_current_values(self):
        self.inputs.update()
        self.summed_inputs[:,:] = self.inputs.get_sum()

        self.current_somatic_voltages[:,:] = self.new_somatic_voltages
        self.current_spiked_neurons[:,:] = self.new_spiked_neurons


class Circuit_Equation_Integrate_and_Fire_Soma(Base_Integrate_and_Fire_Soma):
    def __init__(self, parameter_dict):
        super().__init__(parameter_dict)
        self.set_membrane_function()

    def set_membrane_function(self):
        membrane_function = Circuit_Equation(self.parameters["input_resistance"], self.parameters["membrane_time_constant"], self.summed_inputs)
        self.membrane_solver = RungeKutta2_cupy(membrane_function, self.parameters["time_step"])

    def compute_new_values(self):
        self.time_since_last_spike += self.time_step
        self.new_somatic_voltages += self.membrane_solver.advance(self.current_somatic_voltages, t = 0)

        # set somatic values for neurons that have fired within the refractory period to zero
        #self.new_somatic_voltages *= self.time_since_last_spike > self.refractory_period
        self.set_refractory_values()
        self.new_spiked_neurons[:,:] = self.new_somatic_voltages > self.threshold
        self.reset_spiked_neurons()

        # set this to avoid overlflow
        self.time_since_last_spike = self.cap_array(self.time_since_last_spike, self.upper_limit)


class Izhikevich_Soma(Base_Integrate_and_Fire_Soma):
    def __init__(self, parameter_dict):
        super().__init__(parameter_dict)
        self.current_u = ncp.zeros(self.population_size)
        self.new_u = ncp.zeros(self.population_size)

        self.set_membrane_function()

    def set_membrane_function(self):
        membrane_function = Izhivechik_Equation(self.parameters["membrane_recovery"], self.parameters["resting_potential_variable"], self.summed_inputs, self.population_size)
        self.membrane_solver = RungeKutta2_cupy(membrane_function, self.parameters["time_step"])

    def compute_new_values(self):
        self.time_since_last_spike += self.time_step
        v_u = ncp.concatenate((self.current_somatic_voltages[:,:,ncp.newaxis], self.current_u[:,:,ncp.newaxis]), axis = 2)
        delta_v_u = self.membrane_solver.advance(v_u, t = 0)
        self.new_somatic_voltages += delta_v_u[:,:,0]
        #self.new_somatic_voltages[:,:] = delta_v_u[:,:,0]
        #self.new_somatic_voltages[:,:] = self.current_somatic_voltages
        self.new_u += delta_v_u[:,:,1]



        # set somatic values for neurons that have fired within the refractory period to zero
        #self.new_somatic_voltages *= self.time_since_last_spike > self.refractory_period
        self.set_refractory_values()
        self.new_spiked_neurons[:,:] = self.new_somatic_voltages > self.threshold
        self.new_u += self.new_spiked_neurons*self.parameters["reset_recovery_variable"]



        self.reset_spiked_neurons()

        # destroy values in dead cells
        self.new_somatic_voltages *= self.dead_cells_location
        self.new_spiked_neurons *= self.dead_cells_location

        # set this to avoid overlflow
        self.time_since_last_spike = self.cap_array(self.time_since_last_spike, self.upper_limit)
        self.new_u[:,:] = self.cap_array(self.new_u, self.upper_limit)
        return "Summed inputs", ncp.amax(self.summed_inputs)
    def update_current_values(self):
        super().update_current_values()
        self.current_u[:,:] = self.new_u[:,:]



'''
Axonal arbors
'''
class Dynamical_Axonal_Terminal_Markram_etal_1998(Component):
    interfacable = 0
    def __init__(self, parameter_dict):
        super().__init__(parameter_dict)

        self.spike_matrix = None
        self.delta_t = self.parameters["time_step"]



    def interface(self, external_component):
        # read variable should be a 2d or 3d array containing boolean values of spikes
        self.external_component = external_component
        external_component_read_variable = self.external_component.interfacable
        external_component_read_variable_shape = external_component_read_variable.shape
        self.indexes = self.create_indexes(external_component_read_variable_shape)
        self.spike_matrix = ncp.zeros(external_component_read_variable_shape)

        self.population_size = self.spike_matrix.shape
        self.time_since_last_spike = ncp.zeros(self.population_size)

        if not type(self.parameters["resting_utilization_of_synaptic_efficacy"]) is dict:
            self.resting_utilization_of_synaptic_efficacy = self.parameters["resting_utilization_of_synaptic_efficacy"]

        elif self.parameters["resting_utilization_of_synaptic_efficacy"]["distribution"] == "normal":
            self.resting_utilization_of_synaptic_efficacy = ncp.random.normal(self.parameters["resting_utilization_of_synaptic_efficacy"]["mean"], self.parameters["resting_utilization_of_synaptic_efficacy"]["SD"], self.population_size)

            negative_values = self.resting_utilization_of_synaptic_efficacy <= 0
            replacement_values = ncp.random.uniform(self.parameters["resting_utilization_of_synaptic_efficacy"]["mean"] - self.parameters["resting_utilization_of_synaptic_efficacy"]["SD"], self.parameters["resting_utilization_of_synaptic_efficacy"]["mean"] + self.parameters["resting_utilization_of_synaptic_efficacy"]["SD"], self.population_size)
            self.resting_utilization_of_synaptic_efficacy *= negative_values == 0
            self.resting_utilization_of_synaptic_efficacy += replacement_values*negative_values

        else:
            print("only normal distribution implementee for resting_utilization_of_synaptic_efficacy_distribution")
            sys.exit(0)

        if self.parameters["absolute_synaptic_efficacy"]["distribution"] == "normal":
            self.weight_matrix = ncp.random.normal(self.parameters["absolute_synaptic_efficacy"]["mean"], self.parameters["absolute_synaptic_efficacy"]["SD"], self.population_size)

            if self.parameters["type"] == "excitatory":
                negative_values = self.weight_matrix <= 0
                replacement_values = ncp.random.uniform(self.parameters["absolute_synaptic_efficacy"]["mean"] - self.parameters["absolute_synaptic_efficacy"]["SD"], self.parameters["absolute_synaptic_efficacy"]["mean"] + self.parameters["absolute_synaptic_efficacy"]["SD"], self.population_size)
                self.weight_matrix *= negative_values == 0
                self.weight_matrix += replacement_values*negative_values
            elif self.parameters["type"] == "inhibitory":
                positive_values = self.weight_matrix <= 0
                replacement_values = ncp.random.uniform(self.parameters["absolute_synaptic_efficacy"]["mean"] - self.parameters["absolute_synaptic_efficacy"]["SD"], self.parameters["absolute_synaptic_efficacy"]["mean"] + self.parameters["absolute_synaptic_efficacy"]["SD"], self.population_size)
                self.weight_matrix *= positive_values == 0
                self.weight_matrix += replacement_values*positive_values

        else:
            print("Absolute synaptic efficacy distributions other than normal has not been implemented")
            sys.exit(0)

        if self.parameters["time_constant_depresssion"]["distribution"] == "normal":
            self.tau_recovery = ncp.random.normal(self.parameters["time_constant_depresssion"]["mean"], self.parameters["time_constant_depresssion"]["SD"], self.population_size)

            negative_values = self.tau_recovery <= 0
            replacement_values = ncp.random.uniform(self.parameters["time_constant_depresssion"]["mean"] - self.parameters["time_constant_depresssion"]["SD"], self.parameters["time_constant_depresssion"]["mean"] + self.parameters["time_constant_depresssion"]["SD"], self.population_size)
            self.tau_recovery *= negative_values == 0
            self.tau_recovery += replacement_values*negative_values

        else:
            print("Only normal distribution implemented for time_constant_depression")

        if self.parameters["time_constant_facilitation"]["distribution"] == "normal":
            self.tau_facil = ncp.random.normal(self.parameters["time_constant_facilitation"]["mean"], self.parameters["time_constant_facilitation"]["SD"], self.population_size )

            negative_values = self.tau_facil <= 0
            replacement_values = ncp.random.uniform(self.parameters["time_constant_facilitation"]["mean"] - self.parameters["time_constant_facilitation"]["SD"], self.parameters["time_constant_facilitation"]["mean"] + self.parameters["time_constant_facilitation"]["SD"], self.population_size)
            self.tau_facil *= negative_values == 0
            self.tau_facil += replacement_values*negative_values

        else:
            print("Only normal distribution implemented for time_constant_depression")

        self.current_neurotransmitter_reserve = ncp.ones(self.population_size) # R
        self.new_neurotransmitter_reserve = ncp.ones(self.population_size)

        self.current_utilization_of_synaptic_efficacy = ncp.ones(self.population_size) + self.resting_utilization_of_synaptic_efficacy
        self.new_utilization_of_synaptic_efficacy = ncp.ones(self.population_size)

        self.current_synaptic_response = ncp.zeros(self.population_size)
        self.new_synaptic_response = ncp.zeros(self.population_size)

        if ncp.any(self.tau_recovery <= 0) or ncp.any(self.tau_facil <= 0) or ncp.any(self.resting_utilization_of_synaptic_efficacy <= 0):
            print("unsuccefull at removing negative values")
            sys.exit(0)

        self.interfacable = self.current_synaptic_response

    def compute_new_values(self):


        self.new_utilization_of_synaptic_efficacy[self.indexes] = self.current_utilization_of_synaptic_efficacy * ncp.exp((-self.time_since_last_spike) / self.tau_facil) + self.resting_utilization_of_synaptic_efficacy*(1 - self.current_utilization_of_synaptic_efficacy * ncp.exp((-self.time_since_last_spike) / self.tau_facil))



        self.new_neurotransmitter_reserve[self.indexes] = self.current_neurotransmitter_reserve * (1 - self.new_utilization_of_synaptic_efficacy)*ncp.exp(-self.time_since_last_spike / self.tau_recovery) + 1 - ncp.exp(-self.time_since_last_spike / self.tau_recovery)


        self.time_since_last_spike += self.delta_t
        self.time_since_last_spike *= self.spike_matrix == 0

        self.new_synaptic_response[self.indexes] = self.weight_matrix * self.new_utilization_of_synaptic_efficacy *self.new_neurotransmitter_reserve*self.spike_matrix

        no_spike_mask = self.spike_matrix == 0
        self.new_utilization_of_synaptic_efficacy *= self.spike_matrix
        self.new_utilization_of_synaptic_efficacy += self.current_utilization_of_synaptic_efficacy*no_spike_mask

        self.new_neurotransmitter_reserve *= self.spike_matrix
        self.new_neurotransmitter_reserve += self.current_neurotransmitter_reserve*no_spike_mask


    def update_current_values(self):
        self.current_synaptic_response[self.indexes] = self.new_synaptic_response

        self.current_utilization_of_synaptic_efficacy[self.indexes] = self.new_utilization_of_synaptic_efficacy

        self.current_neurotransmitter_reserve[self.indexes] = self.new_neurotransmitter_reserve

        self.spike_matrix = self.external_component.interfacable

'''
Dendritic spines
'''
class Dendritic_Spine_Maas(Component):
    interfacable = 0
    def __init__(self, parameter_dict):
        super().__init__(parameter_dict)
        self.dt = self.parameters["time_step"]
        self.time_constant = self.parameters["time_constant"]


    def interface(self, external_component):
        self.external_component = external_component
        external_component_read_variable = self.external_component.interfacable
        external_component_read_variable_shape = external_component_read_variable.shape
        self.indexes = self.create_indexes(external_component_read_variable_shape)

        self.current_synaptic_input = ncp.zeros(external_component_read_variable_shape)

        self.population_size = self.current_synaptic_input.shape
        self.last_input_since_spike = ncp.zeros(self.population_size)
        self.new_synaptic_output = ncp.zeros(self.population_size)
        self.current_synaptic_output = ncp.zeros(self.population_size)

        self.time_since_last_spike = ncp.ones(self.population_size) + 1000

        self.interfacable = self.current_synaptic_output
    def compute_new_values(self):
        # compute new time since last spiked first to decay current value
        self.time_since_last_spike += self.dt

        self.new_synaptic_output[self.indexes] = self.last_input_since_spike * ncp.exp(-self.time_since_last_spike / self.time_constant)
        self.new_synaptic_output += self.current_synaptic_input

        current_input_mask = self.current_synaptic_input == 0
        self.last_input_since_spike *= current_input_mask
        self.last_input_since_spike += self.new_synaptic_output * (current_input_mask == 0)

        self.time_since_last_spike *= current_input_mask
        #self.cap_array(self.time_since_last_spike,10000)
        return "max dendritic spine", ncp.amax(self.current_synaptic_input), ncp.amax(self.new_synaptic_output)


    def update_current_values(self):
        self.current_synaptic_output[self.indexes] = self.new_synaptic_output
        self.current_synaptic_input[self.indexes] = self.external_component.interfacable

    def cap_array(self, array, upper_cap):
            below_upper_limit = array < upper_cap
            array *= below_upper_limit
            array += (below_upper_limit == 0)*upper_cap
'''
Arborizers
'''

class Dendritic_Arbor(Component):
    interfacable = 0
    kill_mask = 0
    def __init__(self, parameter_dict):
        super().__init__(parameter_dict)
        self.projection_template = self.parameters["projection_template"]

    def interface(self, external_component):
        self.external_component = external_component
        external_component_read_variable = self.external_component.interfacable
        external_component_read_variable_shape = external_component_read_variable.shape

        # read variable should be a 2d array containing spikes
        self.axonal_hillock_spikes_array = ncp.zeros(external_component_read_variable_shape)


        print("Arborizing axon \n")
        if len(self.projection_template.shape) <= 1:
            print("Projection template has 1 axis")
            self.template_rolls = [[0,0]]
            self.midX = 0
            self.midY = 0
            self.max_level = 1
            if len(self.axonal_hillock_spikes_array.shape) <= 1:
                print("axonal_hillock_spikes_array has 1 axis of size: ", self.axonal_hillock_spikes_array.shape)
                self.new_spike_array = ncp.zeros(self.axonal_hillock_spikes_array.shape[0],dtype='float64')
                self.current_spike_array = ncp.zeros(self.axonal_hillock_spikes_array.shape[0],dtype='float64')
            else:
                print("axonal_hillock_spikes_array has 2 axis of size: ", self.inputs.shape)
                self.new_spike_array = ncp.zeros((self.axonal_hillock_spikes_array.shape[0], self.axonal_hillock_spikes_array.shape[1]) )
                self.current_spike_array = ncp.zeros((self.axonal_hillock_spikes_array.shape[0], self.axonal_hillock_spikes_array.shape[1]) )
        elif len(self.projection_template.shape) == 2:
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
                self.new_spike_array = ncp.zeros((self.axonal_hillock_spikes_array.shape[0], self.max_level))
                self.current_spike_array = ncp.zeros((self.axonal_hillock_spikes_array.shape[0], self.max_level))
            elif (len(self.axonal_hillock_spikes_array.shape) == 2):
                print("axonal_hillock_spikes_array have 2 axis of shape: ", self.axonal_hillock_spikes_array.shape)
                self.new_spike_array = ncp.zeros((self.axonal_hillock_spikes_array.shape[0], self.axonal_hillock_spikes_array.shape[1], self.max_level))
                self.current_spike_array = ncp.zeros((self.axonal_hillock_spikes_array.shape[0], self.axonal_hillock_spikes_array.shape[1], self.max_level))
            else:
                print("######################### \n Error! \n #############################")
                print("axonal_hillock_spikes_array have more than 2 axis: ", self.axonal_hillock_spikes_array.shape)
                sys.exit(1)
            # compute a list that gives the directions a spike should be sent to


        self.population_size = self.current_spike_array.shape

        self.template_rolls = ncp.array(self.template_rolls)

        self.kill_mask = ncp.ones(self.population_size)
        #self.set_boundry_conditions()

        self.interfacable = self.current_spike_array

    def set_boundry_conditions(self):
        if self.parameters["boundry_conditions"] == "closed":
            for index, roll in enumerate(self.template_rolls):
                if roll[0] > 0:
                    self.kill_mask[0:(roll[0]),:,index] = 0
                elif roll[0] < 0:
                    self.kill_mask[(roll[0]):,:,index] = 0
                if roll[1] > 0:
                    self.kill_mask[:,0:(roll[1]), index] = 0
                elif roll[1] < 0:
                    self.kill_mask[:,(roll[1]):, index] = 0


    def kill_connections_based_on_distance(self, base_distance = 0):
        '''
        Base distance is the distance additional to the x,y plane. So for example if you wish to
        create a 3D network you can create two populations, but set the base distance to 1, when
        killing connections between the two layers
        '''

        C = self.parameters["distance_based_connection_probability"]["C"]
        lambda_parameter = self.parameters["distance_based_connection_probability"]["lambda_parameter"]

        nr_of_rolls = self.template_rolls.shape[0]
        base_distances = ncp.ones(nr_of_rolls)
        base_distances = base_distances[:, ncp.newaxis]
        base_distances *= base_distance
        distance_vectors = ncp.concatenate((self.template_rolls, base_distances), axis = 1)
        distance = ncp.linalg.norm(distance_vectors, ord = 2, axis = 1)
        #rhststsngsnrts4 43 2t tewe4t2  2

        random_array = ncp.random.uniform(0,1,self.population_size)

        for distance_index in range(self.population_size[2]):
            self.kill_mask[:,:,distance_index] *= random_array[:,:,distance_index] < C* ncp.exp(-(distance[distance_index]/lambda_parameter)**2)

        return distance

    def compute_new_values(self):
        #print(self.axonal_hillock_spikes_array.shape)
        if self.max_level <= 1:
            self.new_spike_array[:,:] = self.axonal_hillock_spikes_array
        else:
            for i0, x_y in enumerate(self.template_rolls):
                #To do: probably a bad solution to do this in two operations, should try to do it in one

                axonal_hillock_spikes_array_rolled = ncp.roll(self.axonal_hillock_spikes_array, int(x_y[0]), axis = 0)
                axonal_hillock_spikes_array_rolled = ncp.roll(axonal_hillock_spikes_array_rolled, int(x_y[1]), axis = 1)

                self.new_spike_array[:,:,i0] = axonal_hillock_spikes_array_rolled
        self.new_spike_array *= self.kill_mask


    def compute_new_values(self):
        #print(self.axonal_hillock_spikes_array.shape)
        if self.max_level <= 1:
            self.new_spike_array[:,:] = self.axonal_hillock_spikes_array
        else:
            for i0, x_y in enumerate(self.template_rolls):
                #To do: probably a bad solution to do this in two operations, should try to do it in one

                axonal_hillock_spikes_array_rolled = ncp.roll(self.axonal_hillock_spikes_array, int(x_y[0]), axis = 0)
                axonal_hillock_spikes_array_rolled = ncp.roll(axonal_hillock_spikes_array_rolled, int(x_y[1]), axis = 1)

                self.new_spike_array[:,:,i0] = axonal_hillock_spikes_array_rolled
        self.new_spike_array *= self.kill_mask

    def update_current_values(self):
        if self.max_level <= 1:
            self.current_spike_array[:,:] = self.axonal_hillock_spikes_array
        else:
            self.current_spike_array[:,:,:] = self.new_spike_array

        self.axonal_hillock_spikes_array = self.external_component.interfacable

'''
Delay lines
'''
class Delay_Line(Component):
    interfacable = 0
    def __init__(self, parameter_dict):
        super().__init__(parameter_dict)

        self.delay_in_compute_steps = int(self.parameters["delay"] / self.parameters["time_step"])

    def interface(self, external_component):
        self.external_component = external_component
        external_component_read_variable = self.external_component.interfacable
        external_component_read_variable_shape = external_component_read_variable.shape

        # read_variable should be a 2d array of spikes
        self.spike_source = ncp.zeros(external_component_read_variable_shape)

        self.delay_line = ncp.zeros((self.spike_source.shape[0], self.spike_source.shape[1], self.delay_in_compute_steps))
        self.new_spike_output = ncp.zeros(self.spike_source.shape)
        self.current_spike_output = ncp.zeros(self.spike_source.shape)

        self.interfacable = self.current_spike_output

    def compute_new_values(self):
        self.delay_line[:,:,:] = ncp.roll(self.delay_line,1, axis = 2)
        self.new_spike_output[:,:] = self.delay_line[:,:,-1]
        self.delay_line[:,:,0] = self.spike_source
        #return ncp.amax(self.new_spike_output)

    def update_current_values(self):
        self.current_spike_output[:,:] = self.new_spike_output
        self.spike_source = self.external_component.interfacable


'''
Neurons
'''
class Neurons_fully_distributed(object):
    name = ""
    components = {}

    def __init__(self, soma_type, soma_parameter_dict, position, name, client):
        self.client = client

        self.components = {}
        self.name = name
        self.components[self.name] = self.client.submit(soma_type, soma_parameter_dict, actors = True)

        self.connections = []

        self.connected_neurons = {}
        self.position = position



    def interface_futures(self, parameter_dict, neuron):
        '''
        Create the components used in connections between neurons
        The parameter_dict needs to be a dict containing the parameter_dicts of each component
        '''

        self.connected_neurons[neuron.name] = neuron
        connection = []
        if "delay_line" in parameter_dict:


            name = {}
            name["type"] = "delay_line"
            name["ID"] = parameter_dict["delay_line"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = self.client.submit(Delay_Line, parameter_dict["delay_line"], actors = True)

            name = {}
            name["type"] = "arbor"
            name["ID"] = parameter_dict["arbor"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = self.client.submit(Dendritic_Arbor, parameter_dict["arbor"], actors = True)

            name = {}
            name["type"] = "axonal_terminal"
            name["ID"] = parameter_dict["axonal_terminal"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = self.client.submit(Dynamical_Axonal_Terminal_Markram_etal_1998, parameter_dict["axonal_terminal"], actors = True)

            name = {}
            name["type"] = "dendritic_spines"
            name["ID"] = parameter_dict["dendritic_spines"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = self.client.submit(Dendritic_Spine_Maas, parameter_dict["dendritic_spines"], actors = True)

            name = {}
            name["type"] = "soma"
            name["ID"] = neuron.name
            connection.append(name)
            self.connections.append(connection)

        elif "arbor" in parameter_dict:

            name = {}
            name["type"] = "arbor"
            name["ID"] = parameter_dict["arbor"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = self.client.submit(Dendritic_Arbor, parameter_dict["arbor"], actors = True)

            name = {}
            name["type"] = "axonal_terminal"
            name["ID"] = parameter_dict["axonal_terminal"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = self.client.submit(Dynamical_Axonal_Terminal_Markram_etal_1998, parameter_dict["axonal_terminal"], actors = True)

            name = {}
            name["type"] = "dendritic_spines"
            name["ID"] = parameter_dict["dendritic_spines"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = self.client.submit(Dendritic_Spine_Maas, parameter_dict["dendritic_spines"], actors = True)

            name = {}
            name["type"] = "soma"
            name["ID"] = neuron.name
            connection.append(name)
            self.connections.append(connection)

        elif "axonal_terminal" in parameter_dict:
            name = {}
            name["type"] = "axonal_terminal"
            name["ID"] = parameter_dict["axonal_terminal"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = self.client.submit(Dynamical_Axonal_Terminal_Markram_etal_1998, parameter_dict["axonal_terminal"], actors = True)

            name = {}
            name["type"] = "dendritic_spines"
            name["ID"] = parameter_dict["dendritic_spines"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = self.client.submit(Dendritic_Spine_Maas, parameter_dict["dendritic_spines"], actors = True)

            name = {}
            name["type"] = "soma"
            name["ID"] = neuron.name
            connection.append(name)
            self.connections.append(connection)




    def get_component_results(self):
        '''
        Just gets the actual object proxies
        '''
        for key in self.components:
            self.components[key] = self.components[key].result()



    def connect_components(self):
        '''
        This function connects the different components according to the connection sequence defined in interface_future function
        '''
        for connection in self.connections:
            for index, connection_dict in enumerate(connection):
                if index == 0:
                    # If the current component is the first it is interfacing with the neuron's soma
                    somas = self.components[self.name]
                    component = self.components[connection_dict["ID"]]


                    future = component.interface(somas)
                    future.result()

                elif index == len(connection)-1:
                    # if the the current component is the last it is the soma of the neuron this neuron is interfacing with
                    connection_end = self.connected_neurons[connection_dict["ID"]]
                    previous_component_name = connection[index-1]["ID"]
                    previous_component = self.components[previous_component_name]


                    future = connection_end.components[connection_dict["ID"]].interface(previous_component)
                    future.result()

                else:
                    # interface intermediate components
                    component = self.components[connection_dict["ID"]]
                    previous_component_name = connection[index-1]["ID"]
                    previous_component = self.components[previous_component_name]

                    future = component.interface(previous_component)
                    future.result()


                if connection_dict["type"] == "arbor":
                    future = component.set_boundry_conditions()
                    future.result()
                    connection_end_ID = connection[-1]["ID"]
                    connection_end = self.connected_neurons[connection_end_ID]

                    future = component.kill_connections_based_on_distance(self.position - connection_end.position)
                    future.result()



    def compute_new_values(self):
        self.futures = []
        for key in self.components:
            future = self.components[key].compute_new_values()
            self.futures.append(future)

    def update_current_values(self):
        self.futures = []
        for key in self.components:
            future = self.components[key].update_current_values()
            self.futures.append(future)

    def get_results(self):
        for future in self.futures:
            future.result()
        self.futures = []

class Neurons_at_worker(object):
    name = ""
    components = {}
    def __init__(self, soma_type, soma_parameter_dict, position, name):
        #self.client = client

        self.components = {}
        self.name = name
        self.components[self.name] = soma_type(soma_parameter_dict)

        self.connections = []

        self.connected_neurons = {}
        self.position = position



    def interface_futures(self, parameter_dict, neuron):
        '''
        Create the components used in connections between neurons
        The parameter_dict needs to be a dict containing the parameter_dicts of each component
        '''

        self.connected_neurons[neuron.name] = neuron
        connection = []
        if "delay_line" in parameter_dict:


            name = {}
            name["type"] = "delay_line"
            name["ID"] = parameter_dict["delay_line"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = Delay_Line(parameter_dict["delay_line"])

            name = {}
            name["type"] = "arbor"
            name["ID"] = parameter_dict["arbor"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = Dendritic_Arbor(parameter_dict["arbor"])

            name = {}
            name["type"] = "axonal_terminal"
            name["ID"] = parameter_dict["axonal_terminal"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = Dynamical_Axonal_Terminal_Markram_etal_1998(parameter_dict["axonal_terminal"])

            name = {}
            name["type"] = "dendritic_spines"
            name["ID"] = parameter_dict["dendritic_spines"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = Dendritic_Spine_Maas(parameter_dict["dendritic_spines"])

            name = {}
            name["type"] = "soma"
            name["ID"] = neuron.name
            connection.append(name)
            self.connections.append(connection)

        elif "arbor" in parameter_dict:

            name = {}
            name["type"] = "arbor"
            name["ID"] = parameter_dict["arbor"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = Dendritic_Arbor(parameter_dict["arbor"])

            name = {}
            name["type"] = "axonal_terminal"
            name["ID"] = parameter_dict["axonal_terminal"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = Dynamical_Axonal_Terminal_Markram_etal_1998(parameter_dict["axonal_terminal"])

            name = {}
            name["type"] = "dendritic_spines"
            name["ID"] = parameter_dict["dendritic_spines"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = Dendritic_Spine_Maas(parameter_dict["dendritic_spines"])

            name = {}
            name["type"] = "soma"
            name["ID"] = neuron.name
            connection.append(name)
            self.connections.append(connection)

        elif "axonal_terminal" in parameter_dict:
            name = {}
            name["type"] = "axonal_terminal"
            name["ID"] = parameter_dict["axonal_terminal"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = Dynamical_Axonal_Terminal_Markram_etal_1998(parameter_dict["axonal_terminal"])

            name = {}
            name["type"] = "dendritic_spines"
            name["ID"] = parameter_dict["dendritic_spines"]["ID"]
            connection.append(name)
            self.components[name["ID"]] = Dendritic_Spine_Maas(parameter_dict["dendritic_spines"])

            name = {}
            name["type"] = "soma"
            name["ID"] = neuron.name
            connection.append(name)
            self.connections.append(connection)


    def connect_components(self):
        '''
        This function connects the different components according to the connection sequence defined in interface_future function
        '''
        for connection in self.connections:
            for index, connection_dict in enumerate(connection):
                if index == 0:
                    # If the current component is the first it is interfacing with the neuron's soma
                    somas = self.components[self.name]
                    self.components[connection_dict["ID"]]

                elif index == len(connection)-1:
                    # if the the current component is the last it is the soma of the neuron this neuron is interfacing with
                    connection_end = self.connected_neurons[connection_dict["ID"]]
                    previous_component_name = connection[index-1]["ID"]
                    previous_component = self.components[previous_component_name]

                    connection_end.components[connection_dict["ID"]].interface(previous_component)

                else:
                    # interface intermediate components
                    component = self.components[connection_dict["ID"]]
                    previous_component_name = connection[index-1]["ID"]
                    previous_component = self.components[previous_component_name]

                    component.interface(previous_component)


                if connection_dict["type"] == "arbor":
                    component.set_boundry_conditions()

                    connection_end_ID = connection[-1]["ID"]
                    connection_end = self.connected_neurons[connection_end_ID]

                    component.kill_connections_based_on_distance(self.position - connection_end.position)

    def compute_new_values(self):
        for key in self.components:
            self.components[key].compute_new_values()


    def update_current_values(self):
        for key in self.components:
            self.components[key].update_current_values()


class Input_Neurons(Neurons_fully_distributed):
    def __init__(self, Input_Class, input_parameter_dict, position, name, client):
        self.client = client

        self.components = {}
        self.name = name
        self.components[self.name] = self.client.submit(Input_Class, input_parameter_dict, actors = True)

        self.connections = []

        self.connected_neurons = {}
        self.position = position
    def compute_new_values(self, inputs):
        self.futures = []
        for key in self.components:
            if key == self.name:
                future = self.components[self.name].compute_new_values(inputs)
                self.futures.append(future)
            else:
                future = self.components[key].compute_new_values()
                self.futures.append(future)

'''
Input classes
'''
class Inputs_Distribute_Single_spike(object):
    interfacable = 0
    def __init__(self, parameter_dict):
        self.parameters = parameter_dict
        self.new_inputs = ncp.zeros(self.parameters["population_size"])
        self.current_inputs = ncp.zeros(self.parameters["population_size"])
        self.interfacable = self.current_inputs
        self.input_mask = 1
        self.population_size = self.parameters["population_size"]

        input_mask = ncp.random.uniform(0,1, self.parameters["population_size"]) < self.parameters["percent"]
        self.set_input_mask(input_mask)

    def set_input_mask(self, input_mask):
        self.input_mask = input_mask


    def compute_new_values(self, spike):
        self.new_inputs[:,:] = self.input_mask * spike

    def update_current_values(self):
        self.current_inputs[:,:] = self.new_inputs[:,:]

'''
Readouts
'''

class Readout_P_Delta(object):
    def __init__(self, parameter_dict):
        self.parameters = parameter_dict

        self.nr_of_readout_neurons = self.parameters["nr_of_readout_neurons"]
        self.parallel_perceptron_outputs = ncp.zeros(self.nr_of_readout_neurons)

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
        input_projection = ncp.repeat(self.inputs[:,:,ncp.newaxis], self.nr_of_readout_neurons, axis = 2)
        parallel_perceptron_outputs = ncp.sum(input_projection*self.weights, axis = (0,1))
        return parallel_perceptron_outputs

    def update_weights(self, desired_output):
        self.desired_output = desired_output

        #testing fic
        input_projection = ncp.repeat(self.inputs[:,:,cp.newaxis], self.nr_of_readout_neurons, axis = 2)
        parallel_perceptron_outputs = ncp.sum(input_projection*self.weights, axis = (0,1))
        #self.parallel_perceptron_outputs *= 0.3
        #self.parallel_perceptron_outputs += parallel_perceptron_outputs

        #parallel_perceptron_outputs = self.parallel_perceptron_outputs

        #parallel_perceptron_outputs = self.activation_function()

        # summary rule 1
        parallel_perceptron_output_above_equal_0 = parallel_perceptron_outputs >= 0
        # adding axis and transposing to allow the array to be multiplied with 3d input array correctly
        parallel_perceptron_output_above_equal_0 = parallel_perceptron_output_above_equal_0[ncp.newaxis, ncp.newaxis,:].T.T

        #summary rule 2
        parallel_perceptron_output_below_0 = parallel_perceptron_outputs < 0
        parallel_perceptron_output_below_0 = parallel_perceptron_output_below_0[ncp.newaxis, ncp.newaxis,:].T.T
        # summary rule 3, note: margin is yotta in paper
        parallel_perceptron_output_above_0_below_margin = parallel_perceptron_outputs >= 0
        parallel_perceptron_output_above_0_below_margin *= parallel_perceptron_output_above_0_below_margin < self.margin
        parallel_perceptron_output_above_0_below_margin = parallel_perceptron_output_above_0_below_margin[ncp.newaxis, ncp.newaxis,:].T.T

        # summary rule 4
        parallel_perceptron_output_below_0_above_neg_margin = parallel_perceptron_outputs < 0
        parallel_perceptron_output_below_0_above_neg_margin *= parallel_perceptron_outputs > -1*self.margin
        parallel_perceptron_output_below_0_above_neg_margin = parallel_perceptron_output_below_0_above_neg_margin[ncp.newaxis, ncp.newaxis,:].T.T

        # summary rule 5
        #zeros
        weight_update_direction = ncp.zeros(self.weight_shape)


        population_output = ncp.sum(parallel_perceptron_output_above_equal_0) - ncp.sum(parallel_perceptron_output_below_0)
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
        #weight_update_direction = ncp.random.uniform(-1,1,self.weight_shape)
        weight_update_direction *= self.learning_rate


        weight_bounding = self.weights.reshape(self.weights.shape[0]*self.weights.shape[1], self.weights.shape[2])
        weight_bounding = (ncp.linalg.norm(weight_bounding, ord = 2, axis = 0)**2 - 1)
        #print(weight_bounding)
        weight_bounding = weight_bounding[ncp.newaxis, ncp.newaxis,:].T.T
        weight_bounding *= self.learning_rate
        weight_bounding = self.weights * weight_bounding

        # update weights
        self.weights -= weight_bounding
        self.weights += weight_update_direction

        self.current_population_output = population_output

    def classify(self, image):
        input_projection = ncp.repeat(image[:,:,ncp.newaxis], self.nr_of_readout_neurons, axis = 2)
        parallel_perceptron_outputs = ncp.sum(input_projection*self.weights, axis = (0,1))

        parallel_perceptron_output_above_equal_0 = parallel_perceptron_outputs >= 0
        parallel_perceptron_output_below_0 = parallel_perceptron_outputs < 0


        population_output = ncp.sum(parallel_perceptron_output_above_equal_0) - ncp.sum(parallel_perceptron_output_below_0)
        population_output = self.squashing_function(population_output)

        return population_output

    def interface(self, external_component):
        # read_variable is a 2d array of spikes
        self.external_component = external_component
        external_component_read_variable = self.external_component.interfacable
        external_component_read_variable_shape = external_component_read_variable.shape

        self.inputs = ncp.zeros(external_component_read_variable_shape)

        self.weight_shape = list(self.inputs.shape)
        #print("list weight shape ", self.weight_shape)
        self.weight_shape.append(self.nr_of_readout_neurons)
        #print("appended list weight shape ", self.weight_shape)
        self.weights = ncp.random.uniform(-1,1,self.weight_shape)

    def update_current_values(self):
        self.inputs = self.external_component.interfacable


'''
Help functions
'''
class Unique_ID_Dict_Creator(object):
    def __init__(self, ID_length):
        self.ID_length = ID_length
        self.existing_IDs = []
    def create_unique_ID_dict(self, dictionary):
        dictionary = dictionary.copy()

        letters = string.ascii_lowercase
        ID = "".join(random.choice(letters) for i in range(30))
        while ID in self.existing_IDs:
            ID = "".join(random.choice(letters) for i in range(30))
        dictionary["ID"] = ID
        self.existing_IDs.append(ID)
        return dictionary
