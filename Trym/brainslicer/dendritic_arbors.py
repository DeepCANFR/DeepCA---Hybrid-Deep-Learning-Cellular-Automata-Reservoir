
import numpy as ncp 
from help_functions import remove_neg_values
'''
    Axonal arbors
'''
class DynamicalAxonalTerminalMarkramEtal1998(Component):
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

        
        resting_utilization_of_synaptic_efficacy_distribution = self.parameters["resting_utilization_of_synaptic_efficacy"]["distribution"]
        
        if  resting_utilization_of_synaptic_efficacy_distribution == "normal":
            
            mean = self.parameters["resting_utilization_of_synaptic_efficacy"]["mean"]
            SD = self.parameters["resting_utilization_of_synaptic_efficacy"]["SD"]
            population_size = self.state["population_size"]
            

            self.state["resting_utilization_of_synaptic_efficacy"] = ncp.random.normal(mean, SD, population_size)
            
            resting_utilization_of_synaptic_efficacy = self.state["resting_utilization_of_synaptic_efficacy"]
            #
            negative_values = resting_utilization_of_synaptic_efficacy <= 0
            replacement_values = ncp.random.uniform(mean - SD, mean + SD, population_size)
            resting_utilization_of_synaptic_efficacy *= negative_values == 0
            resting_utilization_of_synaptic_efficacy += replacement_values*negative_values

        else:
            print("only normal distribution implemented for resting_utilization_of_synaptic_efficacy_distribution")
            sys.exit(0)

        
        absolute_synaptic_efficacy_distribution = self.parameters["absolute_synaptic_efficacy"]["distribution"]
        
        if resting_utilization_of_synaptic_efficacy_distribution == "normal":
            
            mean = self.parameters["absolute_synaptic_efficacy"]["mean"]
            SD = self.parameters["absolute_synaptic_efficacy"]["SD"]
            population_size = self.state["population_size"]
            
            self.state["weight_matrix"] = ncp.random.normal(mean, SD, population_size)

            
            synapse_type = self.parameters["synapse_type"]
            weight_matrix = self.state["weight_matrix"]
            #


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


        
        time_constant_depression_distribution = self.parameters["time_constant_depression"]["distribution"]
        
        if time_constant_depression_distribution == "normal":
            
            mean =  self.parameters["time_constant_depression"]["mean"]
            SD = self.parameters["time_constant_depression"]["SD"]
            population_size =  self.state["population_size"]
            
            self.state["tau_recovery"] = ncp.random.normal(mean, SD, population_size)

            
            tau_recovery = self.state["tau_recovery"]
            

            negative_values = tau_recovery <= 0
            replacement_values = ncp.random.uniform(mean - SD, mean + SD, population_size)
            tau_recovery *= negative_values == 0
            tau_recovery += replacement_values*negative_values

        else:
            print("Only normal distribution implemented for time_constant_depression")


        if self.parameters["time_constant_facilitation"]["distribution"] == "normal":
            
            mean = self.parameters["time_constant_facilitation"]["mean"]
            SD = self.parameters["time_constant_facilitation"]["SD"]
            population_size = self.state["population_size"]
            

            self.state["tau_facil"] = ncp.random.normal(mean, SD, population_size )

            
            tau_facil = self.state["tau_facil"]
            

            negative_values = tau_facil <= 0
            replacement_values = ncp.random.uniform(mean - SD, mean + SD, population_size)
            tau_facil *= negative_values == 0
            tau_facil += replacement_values*negative_values

        else:
            print("Only normal distribution implemented for time_constant_depression")

        
        population_size = self.state["population_size"]
        resting_utilization_of_synaptic_efficacy = self.state["resting_utilization_of_synaptic_efficacy"]


        self.state["current_neurotransmitter_reserve"] = ncp.ones(population_size) # R
        self.state["new_neurotransmitter_reserve"] = ncp.ones(population_size)

        self.state["current_utilization_of_synaptic_efficacy"] = ncp.ones(population_size) + resting_utilization_of_synaptic_efficacy
        self.state["new_utilization_of_synaptic_efficacy"] = ncp.ones(population_size)

        self.state["current_synaptic_response"] = ncp.zeros(population_size)
        self.state["new_synaptic_response"] = ncp.zeros(population_size)

        
        tau_recovery = self.state["tau_recovery"]
        tau_facil = self.state["tau_facil"]
        #
        if ncp.any(tau_recovery <= 0) or ncp.any(tau_facil <= 0) or ncp.any(resting_utilization_of_synaptic_efficacy <= 0):
            print("unsuccefull at removing negative values")
            sys.exit(0)

        if synapse_type == "inhibitory" and ncp.any(weight_matrix >= 0):
            print("unsucesfull at removing positive values from inhibitory synapse weights")
            sys.exit(0)
        elif synapse_type == "excitatory" and ncp.any(weight_matrix <= 0):
            print("unsucsefull at removing negative values from excitatory synapse weights")
            sys.exit(0)

        
        self.interfacable = self.state["new_synaptic_response"]
        
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
        

        new_utilization_of_synaptic_efficacy[indexes] = current_utilization_of_synaptic_efficacy * ncp.exp((-time_since_last_spike) / tau_facil) + resting_utilization_of_synaptic_efficacy*(1 - current_utilization_of_synaptic_efficacy * ncp.exp((-time_since_last_spike) / tau_facil))

        
        current_neurotransmitter_reserve = self.state["current_neurotransmitter_reserve"]
        new_neurotransmitter_reserve = self.state["new_neurotransmitter_reserve"]
        tau_recovery = self.state["tau_recovery"]
        
        new_neurotransmitter_reserve[indexes] = current_neurotransmitter_reserve * (1 - new_utilization_of_synaptic_efficacy)*ncp.exp(-time_since_last_spike / tau_recovery) + 1 - ncp.exp(-time_since_last_spike / tau_recovery)

        
        time_step = self.parameters["time_step"]
        spike_matrix = self.state["spike_matrix"]
        new_synaptic_response = self.state["new_synaptic_response"]
        weight_matrix = self.state["weight_matrix"]
        

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
        

        current_synaptic_response[indexes] = new_synaptic_response

        current_utilization_of_synaptic_efficacy[indexes] = new_utilization_of_synaptic_efficacy

        current_neurotransmitter_reserve[indexes] = new_neurotransmitter_reserve

        spike_matrix[indexes] = self.external_component.interfacable
        "update"
        # return 2