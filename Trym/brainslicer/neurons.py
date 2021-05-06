import dask 



'''
Neurons
'''
class NeuronsFullyDistributed(object):
    name = ""
    components = {}
    interfacable = 0

    def __init__(self, client):
        self.client = client


    def construct_neuron(self,soma_type, soma_parameter_dict, position, ID, client ):

        self.components = {}
        self.ID = ID
        self.soma_ID = soma_parameter_dict["ID"]
        self.components[self.soma_ID] = self.client.submit(soma_type, soma_parameter_dict, actors = True)
        self.connections = []

        self.connected_neurons = {}
        self.position = position

    def reconstruct_neuron(self, neuron_data):

        self.ID = neuron_data["ID"]
        self.soma_ID = neuron_data["soma_ID"]
        self.position = neuron_data["position"]
        self.connections = neuron_data["connections"]
        self.components = {}
        component_data = neuron_data["component_data"]
        for component_ID in component_data:

            component_parameters = component_data[component_ID]["parameters"]
            component_ID = component_parameters["ID"]
            component_type = component_parameters["type"]

            self.components[component_ID] = self.client.submit(component_type, component_parameters, actors = True)

        self.get_component_results()
        #self.reconstruct_component_states(neuron_data)
        #self.get_results()

    def set_soma(self, soma_data):
        # Use this to change the soma of the neurons for testing the effect of different somatic compartments on
        # the same network
        original_ID = self.components[self.soma_ID].parameters["ID"]
        original_connected_components = self.components[self.soma_ID].state["connected_components"]

        soma_data[1]["parameters"]["ID"] = original_ID
        soma_data[1]["state"]["connected_components"] = original_connected_components

        soma_type = soma_data[1]["parameters"]["type"]
        soma_parameter_dict = soma_data[1]["parameters"]
        self.components[self.soma_ID] = self.client.submit(soma_type, soma_parameter_dict, actors = True)
        self.components[self.soma_ID] = self.components[self.soma_ID].result()
        self.soma = self.components[self.soma_ID]

        self.components[self.soma_ID].set_state(soma_data[1]["state"])
        self.components[self.soma_ID].set_parameters(soma_data[1]["parameters"])


    def set_connected_neurons(self, connected_neurons):
        self.connected_neurons = connected_neurons

    def reconstruct_component_states(self, neuron_data):

        self.futures = []
        component_data = neuron_data["component_data"]
        for component_ID in component_data:
            #component_parameters = component_data[component_ID]
            #component_ID = component_parameters["ID"]
            component_state = component_data[component_ID]["state"]
            future = self.components[component_ID].set_state(component_state)
            self.futures.append(future)
        self.get_results()


    def interface_futures(self, connection_parameters, neuron):
        '''
        Create the components used in connections between neurons
        The parameter_dict needs to be a OrderedDict containing the parameter_dicts of each component
        '''


        self.connected_neurons[neuron.ID] = neuron
        #self.connection_parameters[neuron.ID] = connection_parameters
        connection = []
        connection.append(self.soma_ID)
        for key in connection_parameters:
            component_parameters = connection_parameters[key]
            component_ID = component_parameters["ID"]
            component_type = component_parameters["type"]
            self.components[component_ID] = self.client.submit(component_type, component_parameters, actors = True)
            connection.append(component_ID)
        connection.append(neuron.ID)
        self.connections.append(connection)


    def get_component_results(self):
        '''
        Just gets the actual object proxies
        '''
        for key in self.components:
            self.components[key] = self.components[key].result()
            if key == self.soma_ID:
                self.soma = self.components[key]

    def reconstruct_connections(self):
        for connection in self.connections:
            connection_end_ID = connection[-1]
            connection_end = self.connected_neurons[connection_end_ID]
            for index, ID, in enumerate(connection):
                #print(index, ID)
                if index == 0:
                    pass
                    #print(ID)
                    #somas = self.components[ID]
                    #component = self.components[ID]
                    #print(ID)
                    #future = component.interface(somas)
                    #future.result()
                    #print(ID)
                elif index == len(connection)-1:
                    previous_component_ID = connection[index - 1]
                    previous_component = self.components[previous_component_ID]

                    future = connection_end.soma.reconstruct_interface(previous_component)
                    future.result()
                else:
                    component = self.components[ID]
                    previous_component_ID = connection[index - 1]
                    previous_component = self.components[previous_component_ID]

                    future = component.reconstruct_interface(previous_component)
                    future.result()




    def connect_components(self):
        '''
        This function connects the different components according to the connection sequence defined in interface_future function
        '''


        for connection in self.connections:
            connection_end_ID = connection[-1]
            connection_end = self.connected_neurons[connection_end_ID]
            for index, ID, in enumerate(connection):
                #print(index, ID)
                if index == 0:
                    pass
                    #print(ID)
                    #somas = self.components[ID]
                    #component = self.components[ID]
                    #print(ID)
                    #future = component.interface(somas)
                    #future.result()
                    #print(ID)
                elif index == len(connection)-1:
                    previous_component_ID = connection[index - 1]
                    previous_component = self.components[previous_component_ID]

                    future = connection_end.soma.interface(previous_component)
                    future.result()
                else:
                    component = self.components[ID]
                    previous_component_ID = connection[index - 1]
                    previous_component = self.components[previous_component_ID]

                    future = component.interface(previous_component)
                    future.result()

                    # The first and last components will always be somas so we don't need to check
                    # components type for these
                    component_type = self.components[ID].parameters["type"]
                    if component_type == Dendritic_Arbor:
                        future = component.set_boundry_conditions()
                        future.result()
                        future = component.kill_connections_based_on_distance(self.position - connection_end.position)
                        future.result()



    def compute_new_values(self):
        self.futures = []
        for key in self.components:
            future = self.components[key].compute_new_values()
            self.futures.append(future)

    def update_current_values(self):
        self.futures = []
        for key in self.components:
            future = self.components[key].update_current_values()
            #future.result()
            self.futures.append(future)

    def get_results(self):
        for future in self.futures:
            future.result()
        self.futures = []

    def compile_data(self):
        neuron_data = {}
        neuron_data["ID"] = self.ID
        neuron_data["soma_ID"] = self.soma_ID
        neuron_data["position"] = self.position
        neuron_data["type"] = type(self)
        neuron_data["connections"] = self.connections
        neuron_data["connected_neurons"] = self.connected_neurons.keys()
        component_data = {}
        for key in self.components:
            #print("this is where its wrong ", component)
            future = self.components[key].compile_data()
            out = future.result()
            ID = out[0]
            component_data[ID] = out[1]
        neuron_data["component_data"] = component_data

        return neuron_data



class InputNeurons(NeuronsFullyDistributed):
    interfacable = 0
    parameters = {}

    def __init__(self, client):
        self.client = client

    def construct_neuron(self, Input_Class, input_parameter_dict, position, ID, client):

        self.components = {}
        self.ID = ID
        self.soma_ID = input_parameter_dict["ID"]
        self.components[self.soma_ID] = self.client.submit(Input_Class, input_parameter_dict, actors = True)

        self.connections = []

        self.connected_neurons = {}
        self.position = 0
    def compute_new_values(self, inputs):
        self.futures = []
        for key in self.components:
            if key == self.soma_ID:
                future = self.components[self.soma_ID].compute_new_values(inputs)
                self.futures.append(future)
            else:
                future = self.components[key].compute_new_values()
                self.futures.append(future)
