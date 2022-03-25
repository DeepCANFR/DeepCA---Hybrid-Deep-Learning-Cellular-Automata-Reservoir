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
                "inputs": ["delay_line"],
                "memories":["v", "u", "spiked_neurons", "summed_inputs"],
            }
        }

    graph_genome = {
        "identifier":"DelayLine_test",
        "nodes":{
            "E_Izhikevich_somas_0":excitatory_Izhikevich_somas_genome,
            "delay_line":delay_line_genome,
        },
        "connections":[
            [
                ["E_Izhikevich_somas_0", "delay_line"],
                ["spiked_neurons", "spike_source"]
            ],
            [
                ["delay_line", "E_Izhikevich_somas_0"],
                ["spike_output", "delay_line"]
            ]
        ]
            
    }


    with Client(n_workers = 2) as client:
        graph = bs.DistributedGraph(client, {"IzhikevichNode":bs.IzhikevichNode, "DelayLineNode":bs.DelayLineNode})

        graph.construct_distributed_graph(graph_genome)

        t0 = time.time()
        for t in range(100):
            graph.increment()
        print(time.time() - t0)

        graph.save_memories("test_DelayLine_test_1")