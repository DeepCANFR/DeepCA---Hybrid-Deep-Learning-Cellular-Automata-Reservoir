import numpy as ncp 
from component import Component 

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