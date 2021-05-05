#print("update")#print("new")# return 2import numpy as np
import cupy as cp
import numpy as ncp
# Note: ncp is supposed to offer the option of using either numpy or cupy as dropin for computations
import dask
import string
import random
import time
from collections import OrderedDict
import sys
from differential_equation_solvers import RungeKutta2_cupy, ForwardEuler_cupy
from membrane_equations import Integrate_and_fire_neuron_membrane_function, Circuit_Equation, Izhivechik_Equation
from spike_generators import Poisson_Spike_Generator
from support_classes import Interfacable_Array


VERSION = "0.0.1"



'''
Component classes
##################################################################
'''

class Component(object):
    interfacable = 0
    component_IDs = []
    parameters = {}
    state = {}
    '''
    This is the base class for all components
    Every component must implement the functions given below
    '''

    def __init__(self, parameter_dict):
        self.parameters = parameter_dict
        self.state = {}
        self.state["connected_components"] = []
        time.sleep(0.5)

    def reconstruct_interface(self, external_component):
        self.external_component = external_component

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

    def compile_data(self):
        data = {"parameters": self.parameters, "state":self.state}
        return [self.parameters["ID"], data]

    def set_state(self, state):
        raise NotImplementedError
    def set_parameters(self, parameters):
        self.parameters = parameters


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

        if len(self.parameters["population_size"]) != 2:
            print("Population size must be size 2 and give population size in x and y dimensions")
            sys.exit(0)


        population_size = self.parameters["population_size"]
        refractory_period = self.parameters["refractory_period"]

        ########################################################################

        self.state["time_since_last_spike"] = ncp.ones(population_size) + refractory_period + 1

        self.state["new_spiked_neurons"] = ncp.zeros(population_size)

        self.state["current_spiked_neurons"] = ncp.zeros(population_size)

        ## needs fixing
        self.inputs = Interfacable_Array(population_size)
        self.state["connected_components"] = []

        self.state["summed_inputs"] = ncp.zeros(population_size)

        #self.membrane_solver.set_initial_condition(self.current_somatic_voltages)

        self.state["dead_cells_location"] = 1
        self.interfacable = self.state["new_spiked_neurons"]
        #self.set_membrane_function()

    def set_state(self, state):
        self.state = state
        self.interfacable = self.state["new_spiked_neurons"]
        self.summed_inputs = self.state["summed_inputs"]
        self.set_membrane_function()

    def set_membrane_function(self):
        raise NotImplementedError
        sys.exit(1)

    def reconstruct_interface(self, external_component):
        self.inputs.interface(external_component)

    def interface(self, external_component):
        self.inputs.interface(external_component)
        self.state["connected_components"].append(external_component.parameters["ID"])

    def set_dead_cells(self,dead_cells_location):
        self.state["dead_cells_location"] = dead_cells_location == 0

    def cap_array(self, array, upper_limit):
        below_upper_limit = array < upper_limit
        array *= below_upper_limit
        array += (below_upper_limit == 0)*upper_limit
        return array

    def reset_spiked_neurons(self):
        new_spiked_neurons = self.state["new_spiked_neurons"]
        time_since_last_spike = self.state["time_since_last_spike"]
        new_somatic_voltages = self.state["new_somatic_voltages"]
        reset_voltage = self.state["reset_voltage"]
        #####################################################################
        non_spike_mask = new_spiked_neurons == 0
        time_since_last_spike *= non_spike_mask

        new_somatic_voltages *= non_spike_mask
        new_somatic_voltages += new_spiked_neurons * reset_voltage

    def kill_dead_values(self):
        new_somatic_voltages = self.state["new_somatic_voltages"]
        dead_cells_location = self.state["dead_cells_location"]
        new_spiked_neurons = self.state["new_spiked_neurons"]
        #####################################################################
        # destroy values in dead cells
        new_somatic_voltages *= dead_cells_location
        new_spiked_neurons *= dead_cells_location
    def set_refractory_values(self):
        new_somatic_voltages = self.state["new_somatic_voltages"]
        time_since_last_spike = self.state["time_since_last_spike"]
        refractory_period = self.parameters["refractory_period"]
        reset_voltage = self.state["reset_voltage"]
        #####################################################################
        # set somatic voltages to the reset value if within refractory period
        new_somatic_voltages *= time_since_last_spike > refractory_period
        new_somatic_voltages += (time_since_last_spike <= refractory_period) * reset_voltage

    def compile_data(self):
        data = {"parameters":self.parameters, "state":self.state}
        return self.parameters["ID"], data

    def compute_new_values(self):
        raise NotImplementedError
        sys.exit(1)


    def update_current_values(self):
        summed_inputs = self.state["summed_inputs"]
        current_somatic_voltages = self.state["current_somatic_voltages"]
        current_spiked_neurons = self.state["current_spiked_neurons"]
        new_somatic_voltages = self.state["new_somatic_voltages"]
        new_spiked_neurons = self.state["new_spiked_neurons"]
        #####################################################################
        self.inputs.update()
        summed_inputs[:,:] = self.inputs.get_sum()
        #print("summed inputs in update_current_values ",ncp.amax(summed_inputs))
        current_somatic_voltages[:,:] = new_somatic_voltages[:,:]
        current_spiked_neurons[:,:] = new_spiked_neurons[:,:]
        #print("update")
        return(2)


class Circuit_Equation_Integrate_and_Fire_Soma(Base_Integrate_and_Fire_Soma):
    def __init__(self, parameter_dict):
        super().__init__(parameter_dict)
        ##
        population_size = self.parameters["population_size"]
        #######################################################################

        if self.parameters["reset_voltage"]["distribution"] == "homogenous":
            self.state["reset_voltage"] = self.parameters["reset_voltage"]["value"]
        elif self.parameters["reset_voltage"]["distribution"] == "normal":
            ##
            mean = self.parameters["reset_voltage"]["mean"]
            SD = self.parameters["reset_voltage"]["SD"]
            ####################################################################
            self.state["reset_voltage"] = ncp.random.normal(mean,SD, population_size )

            ##
            reset_voltage = self.state["reset_voltage"]
            ####################################################################
            if self.parameters["reset_voltage"]["pos_neg_uniformity"] == "positive":
                remove_neg_values(reset_voltage, mean, SD)
        elif self.parameters["reset_voltage"]["distribution"] == "uniform":
            ##
            low = self.parameters["reset_voltage"]["low"]
            high = self.parameters["reset_voltage"]["high"]
            ####################################################################
            self.state["reset_voltage"] = ncp.random.uniform(low, high, population_size)

        elif self.parameters["reset_voltage"]["distribution"] == "Izhikevich":
            self.state["reset_voltage"] = ncp.zeros(population_size)
            self.state["reset_voltage"] +=self.parameters["reset_voltage"]["base_value"]
            ##
            membrane_recovery_multiplier = self.parameters["reset_voltage"]["multiplier_value"]
            ####################################################################

            random_variable = ncp.random.uniform(0,1,population_size)
            random_variable = random_variable**2
            membrane_recovery_variance = membrane_recovery_multiplier * random_variable

            self.state["reset_voltage"] += membrane_recovery_variance


        if self.parameters["membrane_time_constant"]["distribution"] == "homogenous":
            self.state["membrane_time_constant"] = self.parameters["membrane_time_constant"]["value"]
        elif self.parameters["membrane_time_constant"]["distribution"] == "normal":
            ##
            mean = self.parameters["membrane_time_constant"]["mean"]
            SD = self.parameters["membrane_time_constant"]["SD"]
            ####################################################################
            self.state["membrane_time_constant"] = ncp.random.normal(mean,SD, population_size )

            ##
            membrane_time_constant = self.state["membrane_time_constant"]
            ####################################################################
            if self.parameters["membrane_time_constant"]["pos_neg_uniformity"] == "positive":
                remove_neg_values(membrane_time_constant, mean, SD)
        elif self.parameters["membrane_time_constant"]["distribution"] == "uniform":
            ##
            low = self.parameters["membrane_time_constant"]["low"]
            high = self.parameters["membrane_time_constant"]["high"]
            ####################################################################
            self.state["membrane_time_constant"] = ncp.random.uniform(low, high, population_size)
        elif self.parameters["membrane_time_constant"]["distribution"] == "Izhikevich":
            self.state["membrane_time_constant"] = ncp.zeros(population_size)
            self.state["membrane_time_constant"] += self.parameters["membrane_time_constant"]["base_value"]

            ##
            membrane_recovery_multiplier = self.parameters["membrane_time_constant"]["multiplier_value"]
            ####################################################################

            random_variable = ncp.random.uniform(0,1,population_size)
            random_variable = random_variable**2
            membrane_recovery_variance = membrane_recovery_multiplier * random_variable

            self.state["membrane_time_constant"] += membrane_recovery_variance

        if self.parameters["input_resistance"]["distribution"] == "homogenous":
            self.state["input_resistance"] = self.parameters["input_resistance"]["value"]
        elif self.parameters["input_resistance"]["distribution"] == "normal":
            ##
            mean = self.parameters["input_resistance"]["mean"]
            SD = self.parameters["input_resistance"]["SD"]
            ####################################################################
            self.state["input_resistance"] = ncp.random.normal(mean,SD, population_size )

            ##
            input_resistance = self.state["input_resistance"]
            ####################################################################
            if self.parameters["input_resistance"]["pos_neg_uniformity"] == "positive":
                remove_neg_values(input_resistance, mean, SD)
        elif self.parameters["input_resistance"]["distribution"] == "uniform":
            ##
            low = self.parameters["input_resistance"]["low"]
            high = self.parameters["input_resistance"]["high"]
            ####################################################################
            self.state["input_resistance"] = ncp.random.uniform(low, high, population_size)
        elif self.parameters["input_resistance"]["distribution"] == "Izhikevich":
            self.state["input_resistance"] = ncp.zeros(population_size)
            self.state["input_resistance"] += self.parameters["input_resistance"]["base_value"]
            random_variable = ncp.random.uniform(0,1,population_size)
            random_variable = random_variable**2
            membrane_recovery_multiplier = self.parameters["input_resistance"]["multiplier_value"]
            membrane_recovery_variance = membrane_recovery_multiplier * random_variable
            self.state["input_resistance"] += membrane_recovery_variance

        if self.parameters["threshold"]["distribution"] == "homogenous":
            self.state["threshold"] = self.parameters["threshold"]["value"]
        elif self.parameters["threshold"]["distribution"] == "normal":
            mean = self.parameters["threshold"]["mean"]
            SD = self.parameters["threshold"]["SD"]
            self.state["threshold"] = ncp.random.normal(mean,SD, population_size )

            ##
            threshold = self.state["threshold"]
            ####################################################################
            if self.parameters["threshold"]["pos_neg_uniformity"] == "positive":
                remove_neg_values(threshold, mean, SD)
        elif self.parameters["threshold"]["distribution"] == "uniform":
            ##
            low = self.parameters["threshold"]["low"]
            high = self.parameters["threshold"]["high"]
            ####################################################################
            self.state["threshold"] = ncp.random.uniform(low, high, population_size)

        elif self.parameters["threshold"]["distribution"] == "Izhikevich":
            self.state["threshold"] = ncp.zeros(population_size)
            self.state["threshold"] += self.parameters["threshold"]["base_value"]
            random_variable = ncp.random.uniform(0,1,population_size)
            random_variable = random_variable**2
            membrane_recovery_multiplier = self.parameters["threshold"]["multiplier_value"]
            membrane_recovery_variance = membrane_recovery_multiplier * random_variable
            self.state["threshold"] += membrane_recovery_variance

        if self.parameters["background_current"]["distribution"] == "homogenous":
            self.state["background_current"] = self.parameters["background_current"]["value"]
        elif self.parameters["background_current"]["distribution"] == "normal":
            mean = self.parameters["background_current"]["mean"]
            SD = self.parameters["background_current"]["SD"]
            self.state["background_current"] = ncp.random.normal(mean,SD, population_size )

            ##
            background_current = self.state["background_current"]
            ####################################################################
            if self.parameters["background_current"]["pos_neg_uniformity"] == "positive":
                remove_neg_values(background_current, mean, SD)
        elif self.parameters["background_current"]["distribution"] == "uniform":
            ##
            low = self.parameters["background_current"]["low"]
            high = self.parameters["background_current"]["high"]
            ####################################################################
            self.state["background_current"] = ncp.random.uniform(low, high, population_size)
        elif self.parameters["background_current"]["distribution"] == "Izhikevich":
            self.state["background_current"] = ncp.zeros(population_size)
            self.state["background_current"] += self.parameters["background_current"]["base_value"]
            random_variable = ncp.random.uniform(0,1,population_size)
            random_variable = random_variable**2
            background_current_multiplier = self.parameters["background_current"]["multiplier_value"]
            background_current_variance = background_current_multiplier * random_variable
            self.state["background_current"] += background_current_variance
        ##
        reset_voltage = self.state["reset_voltage"]
        ########################################################################

        self.state["current_somatic_voltages"] = ncp.ones(population_size) * reset_voltage
        self.state["new_somatic_voltages"] = ncp.ones(population_size) * reset_voltage


        self.set_membrane_function()

    def set_membrane_function(self):
        input_resistance = self.state["input_resistance"]
        membrane_time_constant = self.state["membrane_time_constant"]
        background_current = self.state["background_current"]
        #######################################################################

        membrane_function = Circuit_Equation(input_resistance, membrane_time_constant, self.summed_inputs, background_current)
        self.membrane_solver = RungeKutta2_cupy(membrane_function, self.parameters["time_step"])

    def compute_new_values(self):
        time_step = self.parameters["time_step"]
        upper_limit = self.parameters["temporal_upper_limit"]

        time_since_last_spike  = self.state["time_since_last_spike"]
        current_somatic_voltages = self.state["current_somatic_voltages"]
        new_somatic_voltages = self.state["new_somatic_voltages"]
        new_spiked_neurons = self.state["new_spiked_neurons"]
        threshold = self.state["threshold"]
        dead_cells_location = self.state["dead_cells_location"]
        #####################################################################

        time_since_last_spike += time_step
        new_somatic_voltages += self.membrane_solver.advance(current_somatic_voltages, t = 0)

        # set somatic values for neurons that have fired within the refractory period to zero
        #self.new_somatic_voltages *= self.time_since_last_spike > self.refractory_period
        self.set_refractory_values()
        new_spiked_neurons[:,:] = new_somatic_voltages > threshold
        self.reset_spiked_neurons()

        new_somatic_voltages *= dead_cells_location
        new_spiked_neurons *= dead_cells_location

        # set this to avoid overlflow
        time_since_last_spike = self.cap_array(time_since_last_spike, upper_limit)
        self.state["time_since_last_spike"] = time_since_last_spike
        #print("new")
        # return 1


class Izhikevich_Soma(Base_Integrate_and_Fire_Soma):
    def __init__(self, parameter_dict):
        super().__init__(parameter_dict)
        population_size = self.parameters["population_size"]
        self.state["current_u"] = ncp.zeros(population_size)
        self.state["new_u"] = ncp.zeros(population_size)

        # check if homogenous distributions for all parameters
        print("Checking distributions")
        if self.parameters["membrane_recovery"]["distribution"] == "homogenous":
            self.state["membrane_recovery"] = self.parameters["membrane_recovery"]["value"]
            print("membrane recovery was homogenous")
        if self.parameters["resting_potential_variable"]["distribution"] == "homogenous":
            self.state["resting_potential_variable"] = self.parameters["resting_potential_variable"]["value"]

        if self.parameters["reset_voltage"]["distribution"] == "homogenous":
            self.state["reset_voltage"] = self.parameters["reset_voltage"]["value"]

        if self.parameters["reset_recovery_variable"]["distribution"] == "homogenous":
            self.state["reset_recovery_variable"] = self.parameters["reset_recovery_variable"]["value"]

        # Check if Izhikevich dependnet (a,b)
        if self.parameters["membrane_recovery"]["distribution"] == "Izhikevich":
            self.state["membrane_recovery"] = ncp.zeros(population_size)
            self.state["membrane_recovery"] = +self.parameters["membrane_recovery"]["base_value"]
            random_variable = ncp.random.uniform(0,1,population_size)
            random_variable = random_variable**2
            membrane_recovery_multiplier = self.parameters["membrane_recovery"]["multiplier_value"]
            membrane_recovery_variance = membrane_recovery_multiplier * random_variable
            self.state["membrane_recovery"] += membrane_recovery_variance

            if  self.parameters["resting_potential_variable"]["distribution"] == "Izhikevich" and self.parameters["resting_potential_variable"]["dependent"] == "membrane_recovery":
                self.state["resting_potential_variable"] = ncp.zeros(population_size)
                self.state["resting_potential_variable"] += self.parameters["resting_potential_variable"]["base_value"]
                #random_variable = ncp.random.uniform(0,1,population_size)

                resting_potential_multiplier = self.parameters["resting_potential_variable"]["multiplier_value"]
                resting_potential_variance = random_variable * resting_potential_multiplier
                self.state["resting_potential_variable"] += resting_potential_variance

        if  self.parameters["resting_potential_variable"]["distribution"] == "Izhikevich" and not (self.parameters["resting_potential_variable"]["dependent"] == "membrane_recovery"):
            self.state["resting_potential_variable"] = ncp.zeros(population_size)
            self.state["resting_potential_variable"] += self.parameters["resting_potential_variable"]["base_value"]
            #random_variable = ncp.random.uniform(0,1,population_size)
            random_variable = ncp.random.uniform(0,1,population_size)
            random_variable = random_variable**2

            resting_potential_multiplier = self.parameters["resting_potential_variable"]["multiplier_value"]
            resting_potential_variance = random_variable * resting_potential_multiplier
            self.state["resting_potential_variable"] += resting_potential_variance

        # check if dependent Izhikevich distribution, (c,d)
        if self.parameters["reset_recovery_variable"]["distribution"] == "Izhikevich":
            self.state["reset_recovery_variable"] = ncp.zeros(population_size)
            self.state["reset_recovery_variable"] += self.parameters["reset_recovery_variable"]["base_value"]
            random_variable = ncp.random.uniform(0,1,population_size)
            random_variable = random_variable**2
            reset_recovery_multiplier = self.parameters["reset_recovery_variable"]["multiplier_value"]
            reset_recovery_variance = reset_recovery_multiplier * random_variable
            self.state["reset_recovery_variable"] += reset_recovery_variance

            if  self.parameters["reset_voltage"]["distribution"] == "Izhikevich" and self.parameters["reset_voltage"]["dependent"] == "reset_recovery_variable":
                self.state["reset_voltage"] = ncp.zeros(population_size)
                self.state["reset_voltage"] += self.parameters["reset_voltage"]["base_value"]
                #random_variable = ncp.random.uniform(0,1,population_size)

                reset_voltage_multiplier = self.parameters["reset_voltage"]["multiplier_value"]
                reset_voltage_variance = random_variable * reset_voltage_multiplier
                self.state["reset_voltage"] += reset_voltage_variance

        if  self.parameters["reset_voltage"]["distribution"] == "Izhikevich" and not (self.parameters["reset_voltage"]["dependent"] == "reset_recovery_variable"):
            self.state["reset_voltage"] = ncp.zeros(population_size)
            self.state["reset_voltage"] += self.parameters["reset_voltage"]["base_value"]
            #random_variable = ncp.random.uniform(0,1,population_size)
            random_variable = ncp.random.uniform(0,1,population_size)
            random_variable = random_variable**2

            reset_voltage_multiplier = self.parameters["reset_voltage"]["multiplier_value"]
            reset_voltage_variance = random_variable * reset_voltage_multiplier
            self.state["reset_voltage"] += reset_voltage_variance




        reset_voltage = self.state["reset_voltage"]
        #######################################################################

        self.state["current_somatic_voltages"] = ncp.ones(population_size) * reset_voltage

        self.state["new_somatic_voltages"] = ncp.ones(population_size) * reset_voltage

        #self.state["current_u"][:,:] = self.state["reset_recovery_variable"] * self.state["current_somatic_voltages"]
        #self.state["new_u"][:,:] = self.state["reset_recovery_variable"] * self.state["current_somatic_voltages"]

        self.set_membrane_function()

    def set_membrane_function(self):
        time_step = self.parameters["time_step"]
        population_size = self.parameters["population_size"]

        summed_inputs = self.state["summed_inputs"]
        membrane_recovery = self.state["membrane_recovery"]
        resting_potential_variable = self.state["resting_potential_variable"]
        #################################################################################


        membrane_function = Izhivechik_Equation(membrane_recovery, resting_potential_variable, summed_inputs, population_size)
        self.membrane_solver = RungeKutta2_cupy(membrane_function, time_step)

    def compute_new_values(self):
        time_step = self.parameters["time_step"]
        threshold = self.parameters["threshold"]
        upper_limit = self.parameters["temporal_upper_limit"]

        current_somatic_voltages = self.state["current_somatic_voltages"]
        current_u = self.state["current_u"]
        dead_cells_location = self.state["dead_cells_location"]
        new_somatic_voltages = self.state["new_somatic_voltages"]
        new_u  = self.state["new_u"]
        new_spiked_neurons = self.state["new_spiked_neurons"]
        summed_inputs = self.state["summed_inputs"]
        time_since_last_spike = self.state["time_since_last_spike"]
        reset_recovery_variable = self.state["reset_recovery_variable"]
        #################################################################################

        time_since_last_spike += time_step

        v_u = ncp.concatenate((current_somatic_voltages[:,:,ncp.newaxis], current_u[:,:,ncp.newaxis]), axis = 2)
        delta_v_u = self.membrane_solver.advance(v_u, t = 0)

        new_somatic_voltages += delta_v_u[:,:,0]
        new_u += delta_v_u[:,:,1]



        # set somatic values for neurons that have fired within the refractory period to zero
        #self.new_somatic_voltages *= self.time_since_last_spike > self.refractory_period
        self.set_refractory_values()
        new_spiked_neurons[:,:] = new_somatic_voltages > threshold
        new_u += new_spiked_neurons*reset_recovery_variable

        self.reset_spiked_neurons()

        # destroy values in dead cells
        new_somatic_voltages *= dead_cells_location
        new_spiked_neurons *= dead_cells_location

        # set this to avoid overlflow
        time_since_last_spike = self.cap_array(time_since_last_spike, upper_limit)
        new_u[:,:] = self.cap_array(new_u, upper_limit)
        #return "Summed inputs", ncp.amax(summed_inputs)
        #print("summed inputs in compute_new_values ",ncp.amax(summed_inputs))
        #print("new somatic voltages in compute_new_values ",ncp.amax(new_somatic_voltages))
        #print("new")
        # return 1

    def update_current_values(self):
        current_u = self.state["current_u"]
        new_u = self.state["new_u"]
        ########################################################################
        super().update_current_values()
        current_u[:,:] = new_u[:,:]
        #print("update")
        # return 2



'''
Axonal arbors
'''
class Dynamical_Axonal_Terminal_Markram_etal_1998(Component):
    interfacable = 0
    def __init__(self, parameter_dict):
        super().__init__(parameter_dict)

        self.spike_matrix = None
        self.state["spike_matrix"] = None
        self.delta_t = self.parameters["time_step"]

        self.state["connected_components"] = []


    def interface(self, external_component):
        # read variable should be a 2d or 3d array containing boolean values of spikes
        self.external_component = external_component
        #print(external_component.parameters.keys())
        self.state["connected_components"].append(external_component.parameters["ID"])

        external_component_read_variable = self.external_component.interfacable
        external_component_read_variable_shape = external_component_read_variable.shape
        #self.indexes = self.create_indexes(external_component_read_variable_shape)
        self.state["indexes"] = self.create_indexes(external_component_read_variable_shape)

        #self.spike_matrix = ncp.zeros(external_component_read_variable_shape)
        self.state["spike_matrix"] = ncp.zeros(external_component_read_variable_shape)

        #self.population_size = self.spike_matrix.shape
        self.state["population_size"] = self.state["spike_matrix"].shape

        #self.time_since_last_spike = ncp.zeros(self.population_size)
        self.state["time_since_last_spike"] = ncp.zeros(self.state["population_size"])

        ##
        resting_utilization_of_synaptic_efficacy_distribution = self.parameters["resting_utilization_of_synaptic_efficacy"]["distribution"]
        ########################################################################
        if  resting_utilization_of_synaptic_efficacy_distribution == "normal":
            ##
            mean = self.parameters["resting_utilization_of_synaptic_efficacy"]["mean"]
            SD = self.parameters["resting_utilization_of_synaptic_efficacy"]["SD"]
            population_size = self.state["population_size"]
            ####################################################################

            self.state["resting_utilization_of_synaptic_efficacy"] = ncp.random.normal(mean, SD, population_size)
            ##
            resting_utilization_of_synaptic_efficacy = self.state["resting_utilization_of_synaptic_efficacy"]
            ###
            negative_values = resting_utilization_of_synaptic_efficacy <= 0
            replacement_values = ncp.random.uniform(mean - SD, mean + SD, population_size)
            resting_utilization_of_synaptic_efficacy *= negative_values == 0
            resting_utilization_of_synaptic_efficacy += replacement_values*negative_values

        else:
            print("only normal distribution implemented for resting_utilization_of_synaptic_efficacy_distribution")
            sys.exit(0)

        ##
        absolute_synaptic_efficacy_distribution = self.parameters["absolute_synaptic_efficacy"]["distribution"]
        ########################################################################
        if resting_utilization_of_synaptic_efficacy_distribution == "normal":
            ##
            mean = self.parameters["absolute_synaptic_efficacy"]["mean"]
            SD = self.parameters["absolute_synaptic_efficacy"]["SD"]
            population_size = self.state["population_size"]
            ####################################################################
            self.state["weight_matrix"] = ncp.random.normal(mean, SD, population_size)

            ##
            synapse_type = self.parameters["synapse_type"]
            weight_matrix = self.state["weight_matrix"]
            ###


            if synapse_type == "excitatory":
                #to do: find better solution for switching values
                negative_values = weight_matrix <= 0
                replacement_values = ncp.random.uniform(mean - mean*0.5, mean + mean*0.5, population_size)
                weight_matrix *= negative_values == 0
                weight_matrix += replacement_values*negative_values
            elif synapse_type == "inhibitory":
                positive_values = weight_matrix >= 0
                replacement_values = ncp.random.uniform(mean + mean*0.5, mean - mean*0.5, population_size)
                weight_matrix *= positive_values == 0
                weight_matrix += replacement_values*positive_values

        else:
            print("Absolute synaptic efficacy distributions other than normal has not been implemented")
            sys.exit(0)


        ##
        time_constant_depression_distribution = self.parameters["time_constant_depression"]["distribution"]
        ########################################################################
        if time_constant_depression_distribution == "normal":
            ##
            mean =  self.parameters["time_constant_depression"]["mean"]
            SD = self.parameters["time_constant_depression"]["SD"]
            population_size =  self.state["population_size"]
            ###
            self.state["tau_recovery"] = ncp.random.normal(mean, SD, population_size)

            ##
            tau_recovery = self.state["tau_recovery"]
            ###

            negative_values = tau_recovery <= 0
            replacement_values = ncp.random.uniform(mean - SD, mean + SD, population_size)
            tau_recovery *= negative_values == 0
            tau_recovery += replacement_values*negative_values

        else:
            print("Only normal distribution implemented for time_constant_depression")


        if self.parameters["time_constant_facilitation"]["distribution"] == "normal":
            ##
            mean = self.parameters["time_constant_facilitation"]["mean"]
            SD = self.parameters["time_constant_facilitation"]["SD"]
            population_size = self.state["population_size"]
            ####################################################################

            self.state["tau_facil"] = ncp.random.normal(mean, SD, population_size )

            ##
            tau_facil = self.state["tau_facil"]
            ###

            negative_values = tau_facil <= 0
            replacement_values = ncp.random.uniform(mean - SD, mean + SD, population_size)
            tau_facil *= negative_values == 0
            tau_facil += replacement_values*negative_values

        else:
            print("Only normal distribution implemented for time_constant_depression")

        ##
        population_size = self.state["population_size"]
        resting_utilization_of_synaptic_efficacy = self.state["resting_utilization_of_synaptic_efficacy"]


        self.state["current_neurotransmitter_reserve"] = ncp.ones(population_size) # R
        self.state["new_neurotransmitter_reserve"] = ncp.ones(population_size)

        self.state["current_utilization_of_synaptic_efficacy"] = ncp.ones(population_size) + resting_utilization_of_synaptic_efficacy
        self.state["new_utilization_of_synaptic_efficacy"] = ncp.ones(population_size)

        self.state["current_synaptic_response"] = ncp.zeros(population_size)
        self.state["new_synaptic_response"] = ncp.zeros(population_size)

        ##
        tau_recovery = self.state["tau_recovery"]
        tau_facil = self.state["tau_facil"]
        ###
        if ncp.any(tau_recovery <= 0) or ncp.any(tau_facil <= 0) or ncp.any(resting_utilization_of_synaptic_efficacy <= 0):
            print("unsuccefull at removing negative values")
            sys.exit(0)

        if synapse_type == "inhibitory" and ncp.any(weight_matrix >= 0):
            print("unsucesfull at removing positive values from inhibitory synapse weights")
            sys.exit(0)
        elif synapse_type == "excitatory" and ncp.any(weight_matrix <= 0):
            print("unsucsefull at removing negative values from excitatory synapse weights")
            sys.exit(0)

        ##
        self.interfacable = self.state["new_synaptic_response"]
        ##
    def set_state(self, state):
        self.state = state
        self.interfacable = self.state["new_synaptic_response"]

    def compute_new_values(self):
        indexes = self.state["indexes"]
        current_utilization_of_synaptic_efficacy = self.state["current_utilization_of_synaptic_efficacy"]
        new_utilization_of_synaptic_efficacy = self.state["new_utilization_of_synaptic_efficacy"]
        resting_utilization_of_synaptic_efficacy = self.state["resting_utilization_of_synaptic_efficacy"]
        time_since_last_spike = self.state["time_since_last_spike"]
        tau_facil = self.state["tau_facil"]
        ########################################################################

        new_utilization_of_synaptic_efficacy[indexes] = current_utilization_of_synaptic_efficacy * ncp.exp((-time_since_last_spike) / tau_facil) + resting_utilization_of_synaptic_efficacy*(1 - current_utilization_of_synaptic_efficacy * ncp.exp((-time_since_last_spike) / tau_facil))

        ##
        current_neurotransmitter_reserve = self.state["current_neurotransmitter_reserve"]
        new_neurotransmitter_reserve = self.state["new_neurotransmitter_reserve"]
        tau_recovery = self.state["tau_recovery"]
        ###
        new_neurotransmitter_reserve[indexes] = current_neurotransmitter_reserve * (1 - new_utilization_of_synaptic_efficacy)*ncp.exp(-time_since_last_spike / tau_recovery) + 1 - ncp.exp(-time_since_last_spike / tau_recovery)

        ##
        time_step = self.parameters["time_step"]
        spike_matrix = self.state["spike_matrix"]
        new_synaptic_response = self.state["new_synaptic_response"]
        weight_matrix = self.state["weight_matrix"]
        ###

        time_since_last_spike += time_step
        time_since_last_spike *= spike_matrix == 0


        new_synaptic_response[indexes] = weight_matrix * new_utilization_of_synaptic_efficacy *new_neurotransmitter_reserve*spike_matrix

        no_spike_mask = spike_matrix == 0
        new_utilization_of_synaptic_efficacy *= spike_matrix
        new_utilization_of_synaptic_efficacy += current_utilization_of_synaptic_efficacy*no_spike_mask

        new_neurotransmitter_reserve *= spike_matrix
        new_neurotransmitter_reserve += current_neurotransmitter_reserve*no_spike_mask
        #print(ncp.amax(self.interfacable))
        #print("new")
        # return 1

    def update_current_values(self):
        indexes  = self.state["indexes"]
        current_synaptic_response = self.state["current_synaptic_response"]
        new_synaptic_response = self.state["new_synaptic_response"]
        current_utilization_of_synaptic_efficacy = self.state["current_utilization_of_synaptic_efficacy"]
        new_utilization_of_synaptic_efficacy = self.state["new_utilization_of_synaptic_efficacy"]
        current_neurotransmitter_reserve = self.state["current_neurotransmitter_reserve"]
        new_neurotransmitter_reserve = self.state["new_neurotransmitter_reserve"]
        spike_matrix = self.state["spike_matrix"]
        ########################################################################

        current_synaptic_response[indexes] = new_synaptic_response

        current_utilization_of_synaptic_efficacy[indexes] = new_utilization_of_synaptic_efficacy

        current_neurotransmitter_reserve[indexes] = new_neurotransmitter_reserve

        spike_matrix[indexes] = self.external_component.interfacable
        "update"
        # return 2
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
        self.state["connected_components"].append(external_component.parameters["ID"])

        external_component_read_variable = self.external_component.interfacable
        external_component_read_variable_shape = external_component_read_variable.shape
        self.state["indexes"] = self.create_indexes(external_component_read_variable_shape)

        self.state["current_synaptic_input"] = ncp.zeros(external_component_read_variable_shape)
        current_synaptic_input = self.state["current_synaptic_input"]

        self.state["population_size"] = current_synaptic_input.shape
        population_size = self.state["population_size"]
        self.state["last_input_since_spike"] = ncp.zeros(population_size)
        self.state["new_synaptic_output"] = ncp.zeros(population_size)
        self.state["current_synaptic_output"] = ncp.zeros(population_size)

        self.state["time_since_last_spike"] = ncp.ones(population_size) + 1000

        self.interfacable = self.state["new_synaptic_output"]

    def set_state(self, state):
        self.state = state
        self.interfacable = self.state["new_synaptic_output"]

    def compute_new_values(self):
        indexes = self.state["indexes"]
        time_step = self.parameters["time_step"]
        time_since_last_spike = self.state["time_since_last_spike"]
        new_synaptic_output = self.state["new_synaptic_output"]
        current_synaptic_input = self.state["current_synaptic_input"]
        last_input_since_spike = self.state["last_input_since_spike"]
        time_constant = self.parameters["time_constant"]
        ########################################################################
        # compute new time since last spiked first to decay current value
        time_since_last_spike += time_step

        new_synaptic_output[indexes] = last_input_since_spike * ncp.exp(-time_since_last_spike / time_constant)
        new_synaptic_output += current_synaptic_input

        current_input_mask = current_synaptic_input == 0
        last_input_since_spike *= current_input_mask
        last_input_since_spike += new_synaptic_output * (current_input_mask == 0)

        time_since_last_spike *= current_input_mask
        #self.cap_array(self.time_since_last_spike,10000)
        #print(ncp.amax(self.interfacable))
        #return "max dendritic spine", ncp.amax(current_synaptic_input), ncp.amax(new_synaptic_output)
        #print("new")
        # return 1

    def update_current_values(self):
        current_synaptic_output = self.state["current_synaptic_output"]
        current_synaptic_input = self.state["current_synaptic_input"]
        new_synaptic_output = self.state["new_synaptic_output"]
        indexes = self.state["indexes"]
        ########################################################################

        current_synaptic_output[indexes] = new_synaptic_output
        current_synaptic_input[indexes] = self.external_component.interfacable
        #print("update")
        # return 2

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
        #self.projection_template = self.parameters["projection_template"]
        self.state["connected_components"] = []


    def interface(self, external_component):
        self.external_component = external_component
        external_component_read_variable = self.external_component.interfacable
        external_component_read_variable_shape = external_component_read_variable.shape
        self.state["connected_components"].append(external_component.parameters["ID"])
        # read variable should be a 2d array containing spikes
        self.state["axonal_hillock_spikes_array"] = ncp.zeros(external_component_read_variable_shape)

        ##
        projection_template = self.parameters["projection_template"]
        axonal_hillock_spikes_array = self.state["axonal_hillock_spikes_array"]

        print("Arborizing axon \n")
        if len(projection_template.shape) <= 1:
            print("Projection template has 1 axis")
            template_rolls = [[0,0]]
            midX = 0
            midY = 0
            max_level = 1
            if len(axonal_hillock_spikes_array.shape) <= 1:
                print("axonal_hillock_spikes_array has 1 axis of size: ", axonal_hillock_spikes_array.shape)
                new_spike_array = ncp.zeros(axonal_hillock_spikes_array.shape[0],dtype='float64')
                current_spike_array = ncp.zeros(axonal_hillock_spikes_array.shape[0],dtype='float64')
            else:
                print("axonal_hillock_spikes_array has 2 axis of size: ", self.inputs.shape)
                new_spike_array = ncp.zeros((axonal_hillock_spikes_array.shape[0], axonal_hillock_spikes_array.shape[1]) )
                current_spike_array = ncp.zeros((axonal_hillock_spikes_array.shape[0], axonal_hillock_spikes_array.shape[1]) )
        elif len(projection_template.shape) == 2:
            print("Neihbourhood_template has 2 axis: \n ###################### \n", projection_template)
            print("######################")
            midX = int(-projection_template.shape[0]/2)
            midY = int(-projection_template.shape[1]/2)

            template_rolls = []
            max_level = 0
            for i0 in range(projection_template.shape[0]):
                for i1 in range(projection_template.shape[1]):
                    if projection_template[i0,i1] == 1:
                        template_rolls.append([midX + i0, midY + i1])
                        max_level += 1

            if len(axonal_hillock_spikes_array.shape) <= 1:
                print("axonal_hillock_spikes_array have 1 axis of length: ", axonal_hillock_spikes_array.shape)
                new_spike_array = ncp.zeros((axonal_hillock_spikes_array.shape[0], max_level))
                current_spike_array = ncp.zeros((axonal_hillock_spikes_array.shape[0], max_level))
            elif (len(axonal_hillock_spikes_array.shape) == 2):
                print("axonal_hillock_spikes_array have 2 axis of shape: ", axonal_hillock_spikes_array.shape)
                new_spike_array = ncp.zeros((axonal_hillock_spikes_array.shape[0], axonal_hillock_spikes_array.shape[1], max_level))
                current_spike_array = ncp.zeros((axonal_hillock_spikes_array.shape[0], axonal_hillock_spikes_array.shape[1], max_level))
            else:
                print("######################### \n Error! \n #############################")
                print("axonal_hillock_spikes_array have more than 2 axis: ", axonal_hillock_spikes_array.shape)
                sys.exit(1)
            # compute a list that gives the directions a spike should be sent to

        self.state["new_spike_array"] = new_spike_array
        self.state["current_spike_array"] = current_spike_array
        self.state["population_size"] = current_spike_array.shape
        self.state["midX"] = midX
        self.state["midY"] = midY
        self.state["max_level"] = max_level
        self.state["template_rolls"] = ncp.array(template_rolls)
        self.state["population_size"] = current_spike_array.shape
        population_size = self.state["population_size"]
        self.state["kill_mask"] = ncp.ones(population_size)

        self.interfacable = self.state["new_spike_array"]

    def set_state(self, state):
        self.state = state
        self.interfacable = self.state["new_spike_array"]

    def set_boundry_conditions(self):
        kill_mask = self.state["kill_mask"]
        template_rolls = self.state["template_rolls"]
        boundry_conditions = self.parameters["boundry_conditions"]
        ########################################################################
        if boundry_conditions == "closed":
            for index, roll in enumerate(template_rolls):
                if roll[0] > 0:
                    kill_mask[0:(roll[0]),:,index] = 0
                elif roll[0] < 0:
                    kill_mask[(roll[0]):,:,index] = 0
                if roll[1] > 0:
                    kill_mask[:,0:(roll[1]), index] = 0
                elif roll[1] < 0:
                    kill_mask[:,(roll[1]):, index] = 0


    def kill_connections_based_on_distance(self, base_distance = 0):
        '''
        Base distance is the distance additional to the x,y plane. So for example if you wish to
        create a 3D network you can create two populations, but set the base distance to 1, when
        killing connections between the two layers
        '''
        template_rolls = self.state["template_rolls"]
        population_size = self.state["population_size"]
        kill_mask = self.state["kill_mask"]
        C = self.parameters["distance_based_connection_probability"]["C"]
        lambda_parameter = self.parameters["distance_based_connection_probability"]["lambda_parameter"]
        ########################################################################

        nr_of_rolls = template_rolls.shape[0]
        base_distances = ncp.ones(nr_of_rolls)
        base_distances = base_distances[:, ncp.newaxis]
        base_distances *= base_distance
        distance_vectors = ncp.concatenate((template_rolls, base_distances), axis = 1)
        distance = ncp.linalg.norm(distance_vectors, ord = 2, axis = 1)
        #rhststsngsnrts4 43 2t tewe4t2  2

        random_array = ncp.random.uniform(0,1,population_size)

        for distance_index in range(population_size[2]):
            kill_mask[:,:,distance_index] *= random_array[:,:,distance_index] < C* ncp.exp(-(distance[distance_index]/lambda_parameter)**2)

        return distance

    def compute_new_values(self):
        max_level = self.state["max_level"]
        new_spike_array = self.state["new_spike_array"]
        axonal_hillock_spikes_array = self.state["axonal_hillock_spikes_array"]
        template_rolls  = self.state["template_rolls"]
        kill_mask = self.state["kill_mask"]
        new_spike_array = self.state["new_spike_array"]
        ########################################################################

        if max_level <= 1:
            new_spike_array[:,:] = axonal_hillock_spikes_array[:,:]
        else:
            for i0, x_y in enumerate(template_rolls):
                #To do: probably a bad solution to do this in two operations, should try to do it in one

                axonal_hillock_spikes_array_rolled = ncp.roll(axonal_hillock_spikes_array, (int(x_y[0]), int(x_y[1])), axis = (0,1))
                #axonal_hillock_spikes_array_rolled = ncp.roll(axonal_hillock_spikes_array_rolled, int(x_y[1]), axis = 1)

                new_spike_array[:,:,i0] = axonal_hillock_spikes_array_rolled
        new_spike_array *= kill_mask
        #print("new")
        # return 1

    def update_current_values(self):
        max_level = self.state["max_level"]
        current_spike_array = self.state["current_spike_array"]
        axonal_hillock_spikes_array = self.state["axonal_hillock_spikes_array"]
        new_spike_array = self.state["new_spike_array"]
        ########################################################################

        if max_level <= 1:
            current_spike_array[:,:] = axonal_hillock_spikes_array
        else:
            current_spike_array[:,:,:] = new_spike_array

        axonal_hillock_spikes_array[:,:] = self.external_component.interfacable
        #print("update")
        # return 2
'''
Delay lines
'''
class Delay_Line(Component):
    interfacable = 0
    def __init__(self, parameter_dict):
        super().__init__(parameter_dict)

        self.state["delay_in_compute_steps"] = int(self.parameters["delay"] / self.parameters["time_step"])

    def interface(self, external_component):
        delay_in_compute_steps = self.state["delay_in_compute_steps"]
        ########################################################################

        self.external_component = external_component
        external_component_read_variable = self.external_component.interfacable
        external_component_read_variable_shape = external_component_read_variable.shape

        # read_variable should be a 2d array of spikes
        self.state["spike_source"] = ncp.zeros(external_component_read_variable_shape)
        spike_source = self.state["spike_source"]

        self.state["delay_line"] = ncp.zeros((spike_source.shape[0], spike_source.shape[1], delay_in_compute_steps))
        self.state["new_spike_output"] = ncp.zeros(spike_source.shape)
        self.state["current_spike_output"] = ncp.zeros(spike_source.shape)

        self.interfacable = self.state["new_spike_output"]

    def set_state(self, state):
        self.state = state
        self.interfacable = self.state["new_spike_output"]

    def compute_new_values(self):
        delay_line = self.state["delay_line"]
        new_spike_output = self.state["new_spike_output"]
        spike_source = self.state["spike_source"]
        ########################################################################

        delay_line[:,:,:] = ncp.roll(delay_line,1, axis = 2)
        new_spike_output[:,:] = delay_line[:,:,-1]
        delay_line[:,:,0] = spike_source
        #return ncp.amax(self.new_spike_output)
        #print("new")
        # return 1

    def update_current_values(self):
        current_spike_output = self.state["current_spike_output"]
        new_spike_output = self.state["new_spike_output"]
        spike_source = self.state["spike_source"]
        ########################################################################
        current_spike_output[:,:] = new_spike_output
        spike_source[:,:] = self.external_component.interfacable
        #print("update")
        # return 2


'''
Neurons
'''
class Neurons_fully_distributed(object):
    name = ""
    components = {}
    interfacable = 0

    def __init__(self, client):
        self.client = client


    def construct_neuron(self,soma_type, soma_parameter_dict, position, ID, client ):

        self.components = {}
        self.ID = ID
        self.soma_ID = soma_parameter_dict["ID"]
        self.components[self.soma_ID] = self.client.submit(soma_type, soma_parameter_dict, actors = True)
        self.connections = []

        self.connected_neurons = {}
        self.position = position

    def reconstruct_neuron(self, neuron_data):

        self.ID = neuron_data["ID"]
        self.soma_ID = neuron_data["soma_ID"]
        self.position = neuron_data["position"]
        self.connections = neuron_data["connections"]
        self.components = {}
        component_data = neuron_data["component_data"]
        for component_ID in component_data:

            component_parameters = component_data[component_ID]["parameters"]
            component_ID = component_parameters["ID"]
            component_type = component_parameters["type"]

            self.components[component_ID] = self.client.submit(component_type, component_parameters, actors = True)

        self.get_component_results()
        #self.reconstruct_component_states(neuron_data)
        #self.get_results()

    def set_soma(self, soma_data):
        # Use this to change the soma of the neurons for testing the effect of different somatic compartments on
        # the same network
        original_ID = self.components[self.soma_ID].parameters["ID"]
        original_connected_components = self.components[self.soma_ID].state["connected_components"]

        soma_data[1]["parameters"]["ID"] = original_ID
        soma_data[1]["state"]["connected_components"] = original_connected_components

        soma_type = soma_data[1]["parameters"]["type"]
        soma_parameter_dict = soma_data[1]["parameters"]
        self.components[self.soma_ID] = self.client.submit(soma_type, soma_parameter_dict, actors = True)
        self.components[self.soma_ID] = self.components[self.soma_ID].result()
        self.soma = self.components[self.soma_ID]

        self.components[self.soma_ID].set_state(soma_data[1]["state"])
        self.components[self.soma_ID].set_parameters(soma_data[1]["parameters"])


    def set_connected_neurons(self, connected_neurons):
        self.connected_neurons = connected_neurons

    def reconstruct_component_states(self, neuron_data):

        self.futures = []
        component_data = neuron_data["component_data"]
        for component_ID in component_data:
            #component_parameters = component_data[component_ID]
            #component_ID = component_parameters["ID"]
            component_state = component_data[component_ID]["state"]
            future = self.components[component_ID].set_state(component_state)
            self.futures.append(future)
        self.get_results()


    def interface_futures(self, connection_parameters, neuron):
        '''
        Create the components used in connections between neurons
        The parameter_dict needs to be a OrderedDict containing the parameter_dicts of each component
        '''


        self.connected_neurons[neuron.ID] = neuron
        #self.connection_parameters[neuron.ID] = connection_parameters
        connection = []
        connection.append(self.soma_ID)
        for key in connection_parameters:
            component_parameters = connection_parameters[key]
            component_ID = component_parameters["ID"]
            component_type = component_parameters["type"]
            self.components[component_ID] = self.client.submit(component_type, component_parameters, actors = True)
            connection.append(component_ID)
        connection.append(neuron.ID)
        self.connections.append(connection)


    def get_component_results(self):
        '''
        Just gets the actual object proxies
        '''
        for key in self.components:
            self.components[key] = self.components[key].result()
            if key == self.soma_ID:
                self.soma = self.components[key]

    def reconstruct_connections(self):
        for connection in self.connections:
            connection_end_ID = connection[-1]
            connection_end = self.connected_neurons[connection_end_ID]
            for index, ID, in enumerate(connection):
                #print(index, ID)
                if index == 0:
                    pass
                    #print(ID)
                    #somas = self.components[ID]
                    #component = self.components[ID]
                    #print(ID)
                    #future = component.interface(somas)
                    #future.result()
                    #print(ID)
                elif index == len(connection)-1:
                    previous_component_ID = connection[index - 1]
                    previous_component = self.components[previous_component_ID]

                    future = connection_end.soma.reconstruct_interface(previous_component)
                    future.result()
                else:
                    component = self.components[ID]
                    previous_component_ID = connection[index - 1]
                    previous_component = self.components[previous_component_ID]

                    future = component.reconstruct_interface(previous_component)
                    future.result()




    def connect_components(self):
        '''
        This function connects the different components according to the connection sequence defined in interface_future function
        '''


        for connection in self.connections:
            connection_end_ID = connection[-1]
            connection_end = self.connected_neurons[connection_end_ID]
            for index, ID, in enumerate(connection):
                #print(index, ID)
                if index == 0:
                    pass
                    #print(ID)
                    #somas = self.components[ID]
                    #component = self.components[ID]
                    #print(ID)
                    #future = component.interface(somas)
                    #future.result()
                    #print(ID)
                elif index == len(connection)-1:
                    previous_component_ID = connection[index - 1]
                    previous_component = self.components[previous_component_ID]

                    future = connection_end.soma.interface(previous_component)
                    future.result()
                else:
                    component = self.components[ID]
                    previous_component_ID = connection[index - 1]
                    previous_component = self.components[previous_component_ID]

                    future = component.interface(previous_component)
                    future.result()

                    # The first and last components will always be somas so we don't need to check
                    # components type for these
                    component_type = self.components[ID].parameters["type"]
                    if component_type == Dendritic_Arbor:
                        future = component.set_boundry_conditions()
                        future.result()
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
            #future.result()
            self.futures.append(future)

    def get_results(self):
        for future in self.futures:
            future.result()
        self.futures = []

    def compile_data(self):
        neuron_data = {}
        neuron_data["ID"] = self.ID
        neuron_data["soma_ID"] = self.soma_ID
        neuron_data["position"] = self.position
        neuron_data["type"] = type(self)
        neuron_data["connections"] = self.connections
        neuron_data["connected_neurons"] = self.connected_neurons.keys()
        component_data = {}
        for key in self.components:
            #print("this is where its wrong ", component)
            future = self.components[key].compile_data()
            out = future.result()
            ID = out[0]
            component_data[ID] = out[1]
        neuron_data["component_data"] = component_data

        return neuron_data



class Input_Neurons(Neurons_fully_distributed):
    interfacable = 0
    parameters = {}

    def __init__(self, client):
        self.client = client

    def construct_neuron(self, Input_Class, input_parameter_dict, position, ID, client):

        self.components = {}
        self.ID = ID
        self.soma_ID = input_parameter_dict["ID"]
        self.components[self.soma_ID] = self.client.submit(Input_Class, input_parameter_dict, actors = True)

        self.connections = []

        self.connected_neurons = {}
        self.position = 0
    def compute_new_values(self, inputs):
        self.futures = []
        for key in self.components:
            if key == self.soma_ID:
                future = self.components[self.soma_ID].compute_new_values(inputs)
                self.futures.append(future)
            else:
                future = self.components[key].compute_new_values()
                self.futures.append(future)

'''
Input classes
'''
class Inputs_Distribute_Single_spike(Component):
    parameters = {}
    interfacable = 0
    def __init__(self, parameter_dict):
        super().__init__(parameter_dict)

        population_size = self.parameters["population_size"]
        self.state["population_size"] = population_size
        self.state["new_inputs"] = ncp.zeros(population_size)
        self.state["current_inputs"] = ncp.zeros(population_size)
        self.state["input_mask"] = 1

        if "percent" in self.parameters:
            input_mask = ncp.random.uniform(0,1, population_size) < self.parameters["percent"]
            self.state["input_mask"] = input_mask

        self.interfacable = self.state["new_inputs"]

    def set_state(self, state):
        self.state = state
        self.interfacable = self.state["new_inputs"]

    def set_input_mask(self, input_mask):
        self.state["input_mask"] = input_mask


    def compute_new_values(self, spike):
        new_inputs = self.state["new_inputs"]
        input_mask = self.state["input_mask"]

        new_inputs[:,:] = input_mask * spike
        # return 1

    def update_current_values(self):
        new_inputs = self.state["new_inputs"]
        current_inputs = self.state["current_inputs"]

        current_inputs[:,:] = new_inputs[:,:]
        return 2
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
Reconstructors
'''
class Network(object):
    def __init__(self, network_data):
        self.neurons = {}
        self.components = {}
        for neuron_ID in network_data:
            neuron_data = network_data[neuron_ID]
            soma_parameter_dict = neuron_data["soma_parameters"]
            soma_type = neuron_data["soma_type"]
            position =  neuron_data["position"]
            soma_ID = neuron_data["ID"]
            self.neurons[ID] = Neurons_fully_distributed(soma_type, soma_parameter_dict, position, ID)


        for neuron_ID in network_data:
            neuron_data = network_data[neuron_ID]
            #self.neurons[]

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

def remove_neg_values(array, mean, SD):
    population_size = array.shape
    negative_values = array <= 0
    replacement_values = ncp.random.uniform(mean - SD, mean + SD, population_size)
    array *= negative_values == 0
    array += replacement_values*negative_values


if __name__ == "__main__":
    print(VERSION)