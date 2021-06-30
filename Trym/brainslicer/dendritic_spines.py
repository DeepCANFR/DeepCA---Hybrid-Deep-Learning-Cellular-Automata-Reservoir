import numpy as np
from .neural_structure import NeuralStructureNode
'''
    Dendritic spines
'''
class DendriticSpineMaasNode(NeuralStructureNode):
    def __init__(self, parameters):
        super().__init__(parameters)
        population_size = self.parameters["population_size"]

        # current with next
        self.current_state.update({
            "synaptic_output": np.zeros(population_size),
        })
        self.copy_next_state_from_current_state()

        self.current_state.update({
            "synaptic_input": np.zeros(population_size),
            "time_since_last_spike":np.zeros(population_size) + 1000,
            "last_input_since_spike":np.zeros(population_size)
        })

        time_constant_distribution = self.parameters["time_constant"]
        self.static_state.update({
            "time_constant": self.create_distribution_values(time_constant_distribution, population_size)
        })
    
    def compute_next(self):
        time_step = self.parameters["time_step"]
        synaptic_input = self.current_state["synaptic_input"]
        time_since_last_spike = self.current_state["time_since_last_spike"]
        last_input_since_spike = self.current_state["last_input_since_spike"]
        next_synaptic_output = self.next_state["synaptic_output"]

        time_constant = self.static_state["time_constant"]


        time_since_last_spike += time_step

        np.copyto(
            next_synaptic_output, 
            last_input_since_spike * np.exp(-time_since_last_spike/ time_constant)
        )
        next_synaptic_output += synaptic_input


        current_input_mask = synaptic_input == 0
        last_input_since_spike *= current_input_mask
        last_input_since_spike += next_synaptic_output * (current_input_mask == 0)

        time_since_last_spike *= current_input_mask
        # to avoid overflow for positions that never receive input
        time_since_last_spike[time_since_last_spike>100000] = 100000  # To do: choose a good upper limit

