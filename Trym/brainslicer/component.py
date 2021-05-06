import numpy as ncp

class Component(object):
    '''
    This is the base class for all components
    Every component must implement the functions given below
    '''
    interfacable = 0
    component_IDs = []
    parameters = {}
    state = {}

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
