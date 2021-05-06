
'''
Reconstructors
'''
class Network(object):
    def __init__(self, network_data):
        self.neurons = {}
        self.components = {}
        for neuron_ID in network_data:
            neuron_data = network_data[neuron_ID]
            soma_parameter_dict = neuron_data["soma_parameters"]
            soma_type = neuron_data["soma_type"]
            position =  neuron_data["position"]
            soma_ID = neuron_data["ID"]
            self.neurons[ID] = NeuronsFullyDistributed(soma_type, soma_parameter_dict, position, ID)


        for neuron_ID in network_data:
            neuron_data = network_data[neuron_ID]
            #self.neurons[]
