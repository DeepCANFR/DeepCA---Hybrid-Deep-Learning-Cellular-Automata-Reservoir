
import numpy as np
from .nodes import Node
#import neural_structure
'''
    Arborizers
'''

class ArborizerNode(Node):
    '''
    Arborizers disperses spikes coming from some spike source. It describes the connections 
    between the neurons who sends the spikes and the neurons that receive the spikes. 
    Connections are defined relative to the position of a source neuron. Thus if neuron a
    in a 2d plane is connected to its first left neighbour this connection is defined as (1,0)
    if ut connects to the neuron two position above it this is defined as (0,2). Multiple connections
    are defined in a list of tuples: [(1,0),(0,2)]
    The connections are initially the same for all neurons, but connections can be masked out. In this way
    The connection list given as a parameter to the arborizer serves as a limit for the possible connections
    the arborizer can produce, and these can be pruned away later.
    '''
    def __init__(self, parameters):
        super().__init__(parameters)
        population_size = self.parameters["population_size"]
        connections = self.parameters["connection_relative_position"]
        

        # current states with next
        
        # next states without current states
        nr_of_connections = len(connections)
        connection_array_shape = list(population_size)
        connection_array_shape.append(nr_of_connections)
        connection_array_shape = tuple([int(i) for i in connection_array_shape])

        self.next_state.update({
            "connection_array": np.zeros(connection_array_shape)
        })

        # static states
        self.static_state.update({
            "connection_kill_mask": np.ones(connection_array_shape)
        })
        # current states without next
        self.current_state["spike_source"] = np.zeros(population_size)

        self.set_boundry_conditions()
        if "distance_based_connection_probability" in self.parameters:
            self.kill_connections_based_on_distance()


    def compute_next(self):
        # ToDo: make work for arbitrary population size

        connections = self.parameters["connection_relative_position"]
        connection_kill_mask = self.static_state["connection_kill_mask"]
        connection_array = self.next_state["connection_array"]
        spike_source = self.current_state["spike_source"]

        for index, connection in enumerate(connections):
            connection_array[:,:, index] = np.roll(spike_source, connection, axis = (0,1))

        connection_array *= connection_kill_mask
        
    def set_boundry_conditions(self):
        # toDo: Make work for arbitrary population size

        connection_kill_mask = self.static_state["connection_kill_mask"]
        connections = self.parameters["connection_relative_position"]
        boundry_conditions = self.parameters["boundry_conditions"]

        if boundry_conditions == "closed":
            for index, connection in enumerate(connections):
                if connection[0] > 0:
                    connection_kill_mask[0:(connection[0]), :, index] = 0
                elif connection[0] < 0:
                    connection_kill_mask[(connection[0]):, :, index] = 0
                if connection[1] > 0:
                    connection_kill_mask[:, 0:(connection[1]), index] = 0
                elif connection[1] < 0:
                    connection_kill_mask[:, (connection[1]):, index] = 0

    def kill_connections_based_on_distance(self, base_distance=0):
        '''
        Base distance is the distance additional to the x,y plane. So for example if you wish to
        create a 3D network you can create two populations, but set the base distance to 1, when
        killing connections between the two layers
        '''
        connections = self.parameters["connection_relative_position"]
        connections = np.array(connections)
        population_size = self.parameters["population_size"]
        connection_kill_mask = self.static_state["connection_kill_mask"]

        C = self.parameters["distance_based_connection_probability"]["C"]
        lambda_parameter = self.parameters["distance_based_connection_probability"]["lambda_parameter"]

        nr_of_connections = len(connections)
        base_distances = np.ones(nr_of_connections)
        base_distances = base_distances[:, np.newaxis]
        base_distances *= base_distance
        distance_vectors = np.concatenate(
                                    (connections, base_distances), 
                                    axis=1
                                    )

        distance = np.linalg.norm(distance_vectors, ord=2, axis=1)
        
        connection_array_size = connection_kill_mask.shape
        random_array = np.random.uniform(0, 1, connection_array_size)
        for distance_index in range(len(connections)):
            connection_kill_mask[:, :, distance_index] *= random_array[:, :, distance_index] < C * \
                np.exp(-(distance[distance_index]/lambda_parameter)**2)
        
