
import numpy as np
import brainslicer as bs
import time

from dask.distributed import Client

if __name__ == "__main__":
    population_size = (2,2)
    time_step = 1

    inhibitory_izhikevich_soma_parameters = {
        "identifier":"I_Izhikevich_somas_0",
        "parameters":{
            "type":"IzhikevichNode",
            "population_size":population_size,  
            "threshold":30,                  # mV
            "refractory_period":0,
            "membrane_time_constant":30,    # ms
            "background_current":0,         # mA
            "time_step":time_step,                # ms
            "temporal_upper_limit":1000,
            "membrane_recovery":{
                "distribution_type":"Izhikevich",
                "distribution_parameters":{
                    "base_value":0.02,
                    "multiplier_value":0.08
                },
                "dependent":"resting_potential",
            },
            "resting_potential":{
                "distribution_type":"Izhikevich",
                "distribution_parameters":{
                    "base_value":0.25,
                    "multiplier_value":-0.05
                },
                "dependency":"membrane_recovery"
            },
            "reset_voltage":{
                "distribution_type":"homogenous",
                "distribution_parameters":{
                    "value":-65
                }
            },
            "reset_recovery_variable":{
                "distribution_type":"homogenous",
                "distribution_parameters":{
                    "value":2
                }
            },
            "inputs": ["E_Izhikevich_somas_0"], # ToDo: there is no error when this mismatches with the names in connections
            "memories":["v", "u", "spiked_neurons", "summed_inputs"]
            }
        
    }

    excitatory_Izhikevich_somas_parameters = {
        "identifier":"E_Izhikevich_somas_0",
        "parameters":{
            "type":"IzhikevichNode",
            "population_size":population_size,  
            "threshold":30,                  # mV
            "refractory_period":0,
            "membrane_time_constant":15,    # ms
            "background_current":30,         # mA
            "time_step":time_step,                # ms
            "temporal_upper_limit":1000,
            "membrane_recovery":{
                "distribution_type":"homogenous",
                "distribution_parameters":{
                    "value": 0.02
                },
            },
            "resting_potential":{
                "distribution_type":"homogenous",
                "distribution_parameters":{
                    "value":0.2
                },
                "dependency":"membrane_recovery"
            },
            "reset_voltage":{
                "distribution_type":"Izhikevich",
                "distribution_parameters":{
                    "base_value": -65.8,
                    "multiplier_value": 15
                },
                "dependent":"reset_recovery_variable"
            },
            "reset_recovery_variable":{
                "distribution_type":"Izhikevich",
                "distribution_parameters":{
                    "base_value":8,
                    "multiplier_value":-6
                },
                "dependency":"reset_voltage"
            },
            "inputs": ["E_Izhikevich_somas_0", "I_Izhikevich_somas_0"],
            "memories":["v", "u", "spiked_neurons", "summed_inputs"],
        }
    }

    graph_genome = {
        "identifier":"E_I_Izhikevich_network",
        "nodes": {
            "E_Izhikevich_somas_0":excitatory_Izhikevich_somas_parameters,
            "I_Izhikevich_somas_0":inhibitory_izhikevich_soma_parameters
            },
        "connections": [
            [
                ["E_Izhikevich_somas_0", "E_Izhikevich_somas_0"],
                ["spiked_neurons", "E_Izhikevich_somas_0"]
            ],
            [
                ["E_Izhikevich_somas_0", "I_Izhikevich_somas_0"],
                ["spiked_neurons", "E_Izhikevich_somas_0"]
            ]
        ]
    }


    with Client(n_workers = 2) as client:
        graph = bs.DistributedGraph(client, {"IzhikevichNode":bs.IzhikevichNode})

        graph.construct_distributed_graph(graph_genome)

        t0 = time.time()
        for t in range(100):
            graph.increment()
        print(time.time() - t0)

        graph.save_memories("test_Izhikevich_soma_test_1")
