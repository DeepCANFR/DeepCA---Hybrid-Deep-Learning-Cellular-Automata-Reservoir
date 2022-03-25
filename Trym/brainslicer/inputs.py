import numpy as np
from .neural_structure import NeuralStructureNode
from .nodes import Node

class StaticInput(NeuralStructureNode):
    def __init__(self, parameters):
        super().__init__(parameters)
        input_location = self.parameters["input_location"]

        self.next_state.update({
            "input":np.load(input_location)
        })
        self.copy_next_state_from_current_state()


    def compute_next(self):
        pass

class DynamicInput(NeuralStructureNode):
    def __init__(self, parameters):
        super().__init__(self, parameters)
        input_location = self.parameters["input_location"]

        self.input_sequence = np.load(input_location)
        self.counter = 0
        input_shape = self.input_sequence.shape[:2]

        self.next_state.update({
            "input":self.input_sequence[:,:, self.counter]
        })

        self.warning_given = False

    def compute_next(self):
        if self.input_sequence.shape[2] < self.counter:
            self.counter += 1

            next_input = self.next_state["input"]
            np.copyto(
                next_input,
                self.input_sequence[:,:,self.counter]
            )
            
        elif not(self.warning_give):
            print("WARNING!: input sequence completed")
            self.warning_given = True
    

class SingleSpikeTrainInput(NeuralStructureNode):
    def __init__(self, parameters):
        super().__init__(self, parameters)
        input_location = self.parameters["input_location"]
        network_input_locations = self.parameters["network_input_locations"]
        population_size = self.parameters["population_size"]
        input_weight_distribution = self.parameters["input_weight"]

        self.input_sequence = np.load(input_location)
        self.counter = 0

        self.next_state.update({
            "input":np.zeros(population_size)
        })

        input = self.next_state["input"]
        input[network_input_locations] = 1
        input_weights = self.create_distribution_values(input_weight_distribution, population_size)
        input *= input_weights

        self.warning_given = False

    def compute_next(self):
        if self.input_sequence.shape[2] < self.counter:
            next_input = self.next_state["input"]
            np.copyto(
                next_input,
                self.input_sequence[self.counter,:,:]
            )
            self.counter += 1
        elif not(self.warning_given):
            print("WARNING!: input sequence completed")
            self.warning_given = True
    