
from .neural_structure import  NeuralStructureNode
import numpy as np

'''
    Axonal arbors
'''

class DynamicalAxonalTerminalMarkramEtal1998Node(NeuralStructureNode):
    def __init__(self, parameters):
        super().__init__(parameters)

        population_size = self.parameters["population_size"] # ToDo: this one is 3d, maybe I should use a different name

        # static states
        resting_utilization_of_synaptic_efficacy_distribution = self.parameters["resting_utilization_of_synaptic_efficacy"]
        absolute_synaptic_efficacy_distribution = self.parameters["absolute_synaptic_efficacy"]
        time_constant_depression_distribution = self.parameters["time_constant_depression"]
        time_constant_facilitation_distribution = self.parameters["time_constant_facilitation"]
        

        self.static_state.update({
            "resting_utilization_of_synaptic_efficacy": self.create_distribution_values(resting_utilization_of_synaptic_efficacy_distribution, population_size),
            "absolute_synaptic_efficacy": self.create_distribution_values(absolute_synaptic_efficacy_distribution, population_size),
            "time_constant_depression": self.create_distribution_values(time_constant_depression_distribution, population_size),
            "time_constant_facilitation": self.create_distribution_values(time_constant_facilitation_distribution, population_size)
        })

        # remove values 
        
        absolute_synaptic_efficacy = self.static_state["absolute_synaptic_efficacy"]
        if self.parameters["synapse_type"] == "excitatory":
            self.remove_negative_or_positive_values(absolute_synaptic_efficacy, absolute_synaptic_efficacy_distribution, "negative")
        elif self.parameters["synapse_type"] == "inhibitory":
            self.remove_negative_or_positive_values(absolute_synaptic_efficacy, absolute_synaptic_efficacy_distribution, "positive")

        resting_utilization_of_synaptic_efficacy = self.static_state["resting_utilization_of_synaptic_efficacy"]
        self.remove_negative_or_positive_values(resting_utilization_of_synaptic_efficacy, resting_utilization_of_synaptic_efficacy_distribution, "negative")
        

        time_constant_depression = self.static_state["time_constant_depression"]
        self.remove_negative_or_positive_values(time_constant_depression, time_constant_depression_distribution, "negative")
       

        time_constant_facilitation = self.static_state["time_constant_facilitation"]
        self.remove_negative_or_positive_values(time_constant_facilitation, time_constant_facilitation_distribution, "negative")


        # current states with next state
        self.current_state.update({
                "utilization_of_synaptic_efficacy": np.ones(population_size) + resting_utilization_of_synaptic_efficacy,
                "neurotransmitter_reserve": np.ones(population_size),
                "synaptic_response": np.zeros(population_size)
        })
        self.copy_next_state_from_current_state()

        # current states without next state
        self.current_state.update({
            "spike_source": np.zeros(population_size),
            "time_since_last_spike":np.zeros(population_size)
        })

    def compute_next(self):
        time_step = self.parameters["time_step"]

        resting_utilization_of_synaptic_efficacy = self.static_state["resting_utilization_of_synaptic_efficacy"]
        time_constant_facilitation = self.static_state["time_constant_facilitation"]
        time_constant_depression = self.static_state["time_constant_depression"]
        absolute_synaptic_efficacy = self.static_state["absolute_synaptic_efficacy"]

        current_utilization_of_synaptic_efficacy = self.current_state["utilization_of_synaptic_efficacy"]
        current_neurotransmitter_reserve = self.current_state["neurotransmitter_reserve"]
        time_since_last_spike = self.current_state["time_since_last_spike"]
        spike_source = self.current_state["spike_source"]

        next_utilization_of_synaptic_efficacy = self.next_state["utilization_of_synaptic_efficacy"]
        next_neurotransmitter_reserve = self.next_state["neurotransmitter_reserve"]
        next_synaptic_response = self.next_state["synaptic_response"]


        utilization_of_synaptic_efficacy = current_utilization_of_synaptic_efficacy * np.exp((-time_since_last_spike) / time_constant_facilitation) + resting_utilization_of_synaptic_efficacy*(1 - current_utilization_of_synaptic_efficacy * np.exp((-time_since_last_spike) / time_constant_facilitation))
        np.copyto(next_utilization_of_synaptic_efficacy, utilization_of_synaptic_efficacy)

        neurotransmitter_reserve = current_neurotransmitter_reserve * (1 - next_utilization_of_synaptic_efficacy) * np.exp(-time_since_last_spike / time_constant_depression) + 1 - np.exp(-time_since_last_spike / time_constant_depression)
        np.copyto(next_neurotransmitter_reserve, neurotransmitter_reserve)

        no_spike_mask = spike_source == 0

        time_since_last_spike += time_step 
        time_since_last_spike *= no_spike_mask
        #To avoid overflow
        time_since_last_spike[time_since_last_spike>1000000] = 1000000 # ToDo: choose better value

        synaptic_response = absolute_synaptic_efficacy * next_utilization_of_synaptic_efficacy * next_neurotransmitter_reserve * spike_source
        np.copyto(next_synaptic_response, synaptic_response)

        next_utilization_of_synaptic_efficacy *= spike_source 
        next_utilization_of_synaptic_efficacy += current_neurotransmitter_reserve * no_spike_mask 

        next_neurotransmitter_reserve *= spike_source 
        next_neurotransmitter_reserve += current_neurotransmitter_reserve * no_spike_mask 



