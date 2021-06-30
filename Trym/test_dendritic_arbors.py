import brainslicer as bs
import time
from dask.distributed import Client

if __name__ == "__main__":
    population_size = (2,2)
    time_step = 0.1 # ms

    delay_line_genome = {
        "identifier":"delay_line",
        "parameters":{
            "type":"DelayLineNode",
            "population_size": population_size,
            "time_step": time_step,
            "delay": 1, # ms
            "memories":["spike_source", "spike_output"]
        }
    }

    excitatory_Izhikevich_somas_genome = {
            "identifier":"E_Izhikevich_somas_0",
            "parameters":{
                "type":"IzhikevichNode",
                "population_size":population_size,  
                "threshold":30,                  # mV
                "refractory_period":0,
                "membrane_time_constant":15,    # ms
                "background_current":300,         # mA
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
                "inputs": ["delay_line"],
                "memories":["v", "u", "spiked_neurons", "summed_inputs"],
            }
        }

 

    arborizer_genome = {
        "identifier": "arborizer",
        "parameters":{
            "type":"ArborizerNode",
            "population_size":population_size,
            "connection_relative_position": [(1,0), (0,0)],
            "boundry_conditions":"open",
            "distance_based_connection_probability":{
                "C":100,
                "lambda_parameter":2
            },
            "memories":["connection_array", "spike_source"]

        }
    }

    axonal_spine_genome = {
        "identifier":"axonal_spine",
        "parameters":{
            "time_step":time_step,
            "type":"DynamicalAxonalTerminalMarkramEtal1998Node",
            "population_size": (population_size[0], population_size[1], 2),
            "synapse_type":"excitatory",
            "resting_utilization_of_synaptic_efficacy": {
                "distribution_type": "normal",
                "distribution_parameters": {
                    "loc":0.5,
                    "scale": 0.5/2
                }
            },
            "time_constant_depression":{
                "distribution_type":"normal",
                "distribution_parameters":{
                    "loc":1.1,
                    "scale":1.1/2
                }
            },
            "time_constant_facilitation":{
                "distribution_type":"normal",
                "distribution_parameters":{
                    "loc":0.5,
                    "scale":0.5/2
                }
            },
            "absolute_synaptic_efficacy":{
                "distribution_type":"normal",
                "distribution_parameters": {
                    "loc":60,
                    "scale":60
                }
            },
            "memories":["synaptic_response"]

        }
    }

    graph_genome = {
        "identifier":"Arborizer_test",
        "nodes":{
            "E_Izhikevich_somas_0":excitatory_Izhikevich_somas_genome,
            "delay_line":delay_line_genome,
            "arborizer":arborizer_genome,
            "axonal_spine": axonal_spine_genome
        },
        "connections":[
            [
                ["E_Izhikevich_somas_0", "delay_line"],
                ["spiked_neurons", "spike_source"]
            ],
            [
                ["delay_line", "arborizer"],
                ["spike_output", "spike_source"]
            ],
            [
                ["arborizer", "axonal_spine"],
                ["connection_array", "spike_source"]
            ]
        ]
            
    }


    with Client(n_workers = 4) as client:
        graph = bs.DistributedGraph(client, {"IzhikevichNode":bs.IzhikevichNode, "DelayLineNode":bs.DelayLineNode, "ArborizerNode":bs.ArborizerNode, "DynamicalAxonalTerminalMarkramEtal1998Node":bs.DynamicalAxonalTerminalMarkramEtal1998Node})

        graph.construct_distributed_graph(graph_genome)

        t0 = time.time()
        for t in range(100):
            graph.increment()
        print(time.time() - t0)

        graph.save_memories("test_DendriticArbors_1")