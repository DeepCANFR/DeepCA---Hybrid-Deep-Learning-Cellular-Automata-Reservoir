import numpy as ncp
from help_functions import remove_neg_values

'''
Dendritic spines
'''
class DendriticSpineMaas(Component):
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