import numpy as ncp
import time

class NeuralStructure(object):

    '''

    This is the base class for all components

    Every component must implement the functions given below

    '''

    parameters: dict
    state: dict
    interfacable = 0
    component_IDs = []


    def __init__(self, initial_state={}, connected_components=[], **kwargs):
        '''
            In order for an object to be a neural structure it must have the following attributes:
            Required parameters:
                -
        '''
        self.__dict__.update(kwargs)
        self.state = initial_state
        self.state["connected_components"] = connected_components

        #time.sleep(0.5)

    def connect(self, compenent):
        '''
            Adds a component to connected components.
        '''
        self.state['connected_components'].append(compenent)


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

        data = {"parameters": self.__dict__, "state":self.state}

        return [self.ID, data]


    def set_state(self, state):

        raise NotImplementedError


    def set_indexes(self,population_size):
        self.indexes = []
        for i in range(len(population_size)):
            self.indexes.append(slice(0,-1,1))


    def set_indexes(self,population_size):
        self.indexes = []
        for i in range(len(population_size)):
            self.indexes.append(slice(0,-1,1))
