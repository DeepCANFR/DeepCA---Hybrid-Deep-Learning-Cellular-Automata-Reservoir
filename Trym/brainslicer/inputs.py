import numpy as ncp
from .neural_structure import NeuralStructure


'''

Input classes

'''



class InputsDistributeSingleSpike(NeuralStructure):


    def __init__(self, **kwargs):
        '''
            if percent is supplied as a parameter ... TODO
        '''

        super().__init__(**kwargs)
        interfacable = 0


        population_size = self.population_size


        self.state = {"population_size": population_size,
                      "new_inputs":ncp.zeros(population_size),
                      "current_inputs": ncp.zeros(population_size),
                      "input_mask": 1}


        if self.percent:
            random_mask = ncp.random.uniform(low=0, high=1,
                                             size=population_size)
            self.state["input_mask"] = random_mask < self.percent
            


        self.interfacable = self.state["new_inputs"]


    def set_state(self, state):
        self.state = state

        self.interfacable = self.state["new_inputs"]


    def set_input_mask(self, input_mask):

        self.state["input_mask"] = input_mask


    def compute_new_values(self, spike):

        new_inputs = self.state["new_inputs"]

        input_mask = self.state["input_mask"]


        new_inputs[:, :] = input_mask * spike

        # return 1


    def update_current_values(self):

        new_inputs = self.state["new_inputs"]
        current_inputs = self.state["current_inputs"]


        current_inputs[:, :] = new_inputs[:, :]

        return 2

