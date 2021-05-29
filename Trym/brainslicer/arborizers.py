
import numpy as ncp
import sys
from .neural_structure import NeuralStructure
#import neural_structure
'''
    Arborizers
'''
class DendriticArbor(NeuralStructure):
    interfacable = 0
    kill_mask = 0
    def __init__(self, projection_template: ncp.ndarray, distance_based_connection_probability: dict, **kwargs):
        '''
            In order to be a dendritic arbor the following attributes must be defined.
            Required parameters:
                projection_template - numpy-like array
                distance_based_connection_probability - dict containing C and lambda parameter

        '''
        super().__init__(**kwargs)
        self.projection_template = projection_template
        self.__dict__.update(**kwargs)
        #self.projection_template = self.projection_template
        self.state["connected_components"] = []
    
    def interface(self, external_component):
        self.external_component = external_component
        external_component_read_variable = self.external_component.interfacable
        external_component_read_variable_shape = external_component_read_variable.shape
        self.state["connected_components"].append(
            external_component.identifier)
        # read variable should be a 2d array containing spikes
        self.state["axonal_hillock_spikes_array"] = ncp.zeros(
            external_component_read_variable_shape)
        
        axonal_hillock_spikes_array = self.state["axonal_hillock_spikes_array"]
        print("Arborizing axon \n")
        shape = self.projection_template.shape
        if len(self.projection_template.shape) <= 1:
            print("Projection template has 1 axis")
            template_rolls = [[0, 0]]
            midX = 0
            midY = 0
            max_level = 1
            if len(axonal_hillock_spikes_array.shape) <= 1:
                print("axonal_hillock_spikes_array has 1 axis of size: ",
                      axonal_hillock_spikes_array.shape)
                new_spike_array = ncp.zeros(
                    axonal_hillock_spikes_array.shape[0], dtype='float64')
                current_spike_array = ncp.zeros(
                    axonal_hillock_spikes_array.shape[0], dtype='float64')
            else:
                print("axonal_hillock_spikes_array has 2 axis of size: ",
                      self.inputs.shape)
                new_spike_array = ncp.zeros(
                    (axonal_hillock_spikes_array.shape[0], axonal_hillock_spikes_array.shape[1]))
                current_spike_array = ncp.zeros(
                    (axonal_hillock_spikes_array.shape[0], axonal_hillock_spikes_array.shape[1]))
        elif len(shape) == 2:
            print("Neihbourhood_template has 2 axis: \n ###################### \n",
                  self.projection_template)
            print("######################")
            midX = int(-shape[0]/2)
            midY = int(-shape[1]/2)
            template_rolls = []
            max_level = 0
            for i0 in range(shape[0]):
                for i1 in range(shape[1]):
                    if self.projection_template[i0, i1] == 1:
                        template_rolls.append([midX + i0, midY + i1])
                        max_level += 1
            if len(axonal_hillock_spikes_array.shape) <= 1:
                print("axonal_hillock_spikes_array have 1 axis of length: ",
                      axonal_hillock_spikes_array.shape)
                new_spike_array = ncp.zeros(
                    (axonal_hillock_spikes_array.shape[0], max_level))
                current_spike_array = ncp.zeros(
                    (axonal_hillock_spikes_array.shape[0], max_level))
            elif (len(axonal_hillock_spikes_array.shape) == 2):
                print("axonal_hillock_spikes_array have 2 axis of shape: ",
                      axonal_hillock_spikes_array.shape)
                new_spike_array = ncp.zeros(
                    (axonal_hillock_spikes_array.shape[0], axonal_hillock_spikes_array.shape[1], max_level))
                current_spike_array = ncp.zeros(
                    (axonal_hillock_spikes_array.shape[0], axonal_hillock_spikes_array.shape[1], max_level))
            else:
                print(
                    "######################### \n Error! \n #############################")
                print("axonal_hillock_spikes_array have more than 2 axis: ",
                      axonal_hillock_spikes_array.shape)
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
        boundry_conditions = self.boundry_conditions
        if boundry_conditions == "closed":
            for index, roll in enumerate(template_rolls):
                if roll[0] > 0:
                    kill_mask[0:(roll[0]), :, index] = 0
                elif roll[0] < 0:
                    kill_mask[(roll[0]):, :, index] = 0
                if roll[1] > 0:
                    kill_mask[:, 0:(roll[1]), index] = 0
                elif roll[1] < 0:
                    kill_mask[:, (roll[1]):, index] = 0
    def kill_connections_based_on_distance(self, base_distance=0):
        '''
            Base distance is the distance additional to the x,y plane. So for example if you wish to
            create a 3D network you can create two populations, but set the base distance to 1, when
            killing connections between the two layers
        '''
        template_rolls = self.state["template_rolls"]
        population_size = self.state["population_size"]
        kill_mask = self.state["kill_mask"]
        C = self.distance_based_connection_probability["C"]
        lambda_parameter = self.distance_based_connection_probability["lambda_parameter"]
        nr_of_rolls = template_rolls.shape[0]
        base_distances = ncp.ones(nr_of_rolls)
        base_distances = base_distances[:, ncp.newaxis]
        base_distances *= base_distance
        distance_vectors = ncp.concatenate(
            (template_rolls, base_distances), axis=1)
        distance = ncp.linalg.norm(distance_vectors, ord=2, axis=1)
        # rhststsngsnrts4 43 2t tewe4t2  2
        random_array = ncp.random.uniform(0, 1, population_size)
        for distance_index in range(population_size[2]):
            kill_mask[:, :, distance_index] *= random_array[:, :, distance_index] < C * \
                ncp.exp(-(distance[distance_index]/lambda_parameter)**2)
        return distance
    def compute_new_values(self):
        max_level = self.state["max_level"]
        new_spike_array = self.state["new_spike_array"]
        axonal_hillock_spikes_array = self.state["axonal_hillock_spikes_array"]
        template_rolls = self.state["template_rolls"]
        kill_mask = self.state["kill_mask"]
        new_spike_array = self.state["new_spike_array"]
        if max_level <= 1:
            new_spike_array[:, :] = axonal_hillock_spikes_array[:, :]
        else:
            for i0, x_y in enumerate(template_rolls):
                # To do: probably a bad solution to do this in two operations, should try to do it in one
                axonal_hillock_spikes_array_rolled = ncp.roll(
                    axonal_hillock_spikes_array, (int(x_y[0]), int(x_y[1])), axis=(0, 1))
                #axonal_hillock_spikes_array_rolled = ncp.roll(axonal_hillock_spikes_array_rolled, int(x_y[1]), axis = 1)
                new_spike_array[:, :, i0] = axonal_hillock_spikes_array_rolled
        new_spike_array *= kill_mask
        # print("new")
        # return 1
    def update_current_values(self):
        max_level = self.state["max_level"]
        current_spike_array = self.state["current_spike_array"]
        axonal_hillock_spikes_array = self.state["axonal_hillock_spikes_array"]
        new_spike_array = self.state["new_spike_array"]
        if max_level <= 1:
            current_spike_array[:, :] = axonal_hillock_spikes_array
        else:
            current_spike_array[:, :, :] = new_spike_array
        axonal_hillock_spikes_array[:,
                                    :] = self.external_component.interfacable
        # print("update")
        # return 2
