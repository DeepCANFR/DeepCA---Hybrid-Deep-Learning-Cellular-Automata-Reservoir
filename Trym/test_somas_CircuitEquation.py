
import numpy as np
import brainslicer as bs
import time

from dask.distributed import Client

if __name__ == "__main__":
    population_size = (2,2)
    time_step = 1

    inhibitory_CircuitEquation_soma_genome = {
        "identifier":"I_CircuitEquation_somas_0",
        "parameters":{
            "type":"CircuitEquationNode",
            "population_size":population_size, 
            "time_step":time_step,  
            "thresholds":{
                "distribution_type": "homogenous",
                "distribution_parameters": {
                    "value":12
                }
            },                 
            "refractory_period":{
                "distribution_type":"homogenous",
                "distribution_parameters":{
                    "value":0
                }
            },
            "membrane_time_constant":{
                "distribution_type": "homogenous",
                "distribution_parameters": {
                    "value":30
                }
            },   
            "background_current":{
                "distribution_type":"homogenous",
                "distribution_parameters":{
                    "value":0
                }       
            },          
            "reset_voltage":{
                "distribution_type":"homogenous",
                "distribution_parameters":{
                    "value":0
                }
            },
            "input_resistance":{
                "distribution_type":"homogenous",
                "distribution_parameters":{
                    "value":1
                }
            },
            "time_constant":{
                "distribution_type":"homogenous",
                "distribution_parameters":{
                    "value":30
                }
            },
            "inputs": ["E_CircuitEquation_somas_0"], # ToDo: there is no error when this mismatches with the names in connections
            "memories":["v", "spiked_neurons"]
            }
        
    }

    excitatory_CircuitEquation_soma_genome = {
        "identifier":"E_CircuitEquation_somas_0",
        "parameters":{
            "type":"CircuitEquationNode",
            "population_size":population_size, 
            "time_step":time_step,  
            "thresholds":{
                "distribution_type": "homogenous",
                "distribution_parameters": {
                    "value":12
                }
            },                 
            "refractory_period":{
                "distribution_type":"homogenous",
                "distribution_parameters":{
                    "value":0
                }
            },
            "membrane_time_constant":{
                "distribution_type": "homogenous",
                "distribution_parameters": {
                    "value":30
                }
            },   
            "background_current":{
                "distribution_type":"homogenous",
                "distribution_parameters":{
                    "value":100
                }       
            },          
            "reset_voltage":{
                "distribution_type":"homogenous",
                "distribution_parameters":{
                    "value":0
                }
            },
            "input_resistance":{
                "distribution_type":"homogenous",
                "distribution_parameters":{
                    "value":1
                }
            },
            "time_constant":{
                "distribution_type":"homogenous",
                "distribution_parameters":{
                    "value":30
                }
            },
            "inputs": ["E_CircuitEquation_somas_0"], # ToDo: there is no error when this mismatches with the names in connections
            "memories":["v", "spiked_neurons"]
            }
        
    }

    graph_genome = {
        "identifier":"E_I_CircuitEquation_network",
        "nodes": {
            "E_CircuitEquation_somas_0":excitatory_CircuitEquation_soma_genome,
            "I_CircuitEquation_somas_0":inhibitory_CircuitEquation_soma_genome
            },
        "connections": [
            [
                ["E_CircuitEquation_somas_0", "E_CircuitEquation_somas_0"],
                ["spiked_neurons", "E_CircuitEquation_somas_0"]
            ],
            [
                ["E_CircuitEquation_somas_0", "I_CircuitEquation_somas_0"],
                ["spiked_neurons", "E_CircuitEquation_somas_0"]
            ]
        ]
    }


    with Client(n_workers = 2) as client:
        graph = bs.DistributedGraph(client, {"CircuitEquationNode":bs.CircuitEquationNode})

        graph.construct_distributed_graph(graph_genome)

        t0 = time.time()
        for t in range(100):
            graph.increment()
        print(time.time() - t0)

        graph.save_memories("test_CircuitEquation_soma_test_1")
