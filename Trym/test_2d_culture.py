import brainslicer as bs
import time
from dask.distributed import Client

if __name__ == "__main__":
    population_size = (500,500)
    time_step = 1 # ms

    input_genome = {
        "identifier":"input",
        "parameters":{
            "type":"StaticInput",
            "input_location":"/home/trymlind/DeepCA---Hybrid-Deep-Learning-Cellular-Automata-Reservoir/Trym/central_square_input.npy"
        }
    }

    excitatory_Izhikevich_somas_genome = {
            "identifier":"soma",
            "parameters":{
                "type":"IzhikevichNode",
                "population_size":population_size,  
                "threshold":30,                  # mV
                "refractory_period":0,
                "membrane_time_constant":15,    # ms
                "background_current":0,         # mA
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
                "inputs": ["static_input", "process_1-dendritic_spine", "process_2-dendritic_spine", "I_soma-process_1-dendritic_spine", "I_soma-process_2-dendritic_spine"],
                "memories":["spiked_neurons"],
            }
        }
    
    EE_delay_line_genome = {
        "identifier":"EE_delay_line",
        "parameters":{
            "type":"DelayLineNode",
            "population_size": population_size,
            "time_step": time_step,
            "delay": 3, # ms
            "memories":[]
        }
    }
    
    EI_IE_delay_line_genome = {
        "identifier":"EI_IE_delay_line",
        "parameters":{
            "type":"DelayLineNode",
            "population_size": population_size,
            "time_step": time_step,
            "delay": 1, # ms
            "memories":[]
        }
    }

    

 
    neighbours_1 = [(1,1),(-1,1),(0,1),(2,2)]
    arborizer_genome = {
        "identifier": "arborizer",
        "parameters":{
            "type":"ArborizerNode",
            "population_size":population_size,
            "connection_relative_position": neighbours_1,
            "boundry_conditions":"open",
            "distance_based_connection_probability":{
                "C":100,
                "lambda_parameter":2
            },
            "memories":[]

        }
    }

    E_axonal_spine_genome = {
        "identifier":"axonal_spine",
        "parameters":{
            "time_step":time_step,
            "type":"DynamicalAxonalTerminalMarkramEtal1998Node",
            "population_size": (population_size[0], population_size[1], len(neighbours_1)),
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
            "memories":[]

        }
    }

    

    dendritic_spine_genome = {
        "identifier":"dendritic_spine",
        "parameters":{
            "population_size": (population_size[0], population_size[1], len(neighbours_1)),
            "time_step": time_step,
            "type":"DendriticSpineMaasNode",
            "time_constant":{
                "distribution_type": "homogenous",
                "distribution_parameters":{
                    "value":3
                }
            },
            "memories":[]

        }
    }

    EE_process_1_genome = {
        "identifier":"EE_process_1",
        "nodes":{
            "arborizer":arborizer_genome,
            "axonal_spine":E_axonal_spine_genome,
            "dendritic_spine":dendritic_spine_genome
        },
        "connections": [
            [
                ["arborizer", "axonal_spine"],
                ["connection_array", "spike_source"]
            ],
            [
                ["axonal_spine", "dendritic_spine"],
                ["synaptic_response", "synaptic_input"]
            ],
        ]
    }

    EE_process_2_genome = bs.copy_graph_genome(EE_process_1_genome, "EE_process_2")
    
    neighbours_2 = [(1,2),(0,2),(-1,2),(-2,2)]
    EE_process_2_genome["nodes"]["arborizer"]["parameters"]["connection_relative_position"] = neighbours_2

    EI_process_1_genome = bs.copy_graph_genome(EE_process_1_genome, "EI_process_1")
    EI_process_2_genome = bs.copy_graph_genome(EE_process_2_genome, "EI_process_2")


    excitatory_neuron_genome = {
        "identifier":"excitatory_neuron",
        "nodes":{
            "input":input_genome,
            "soma":excitatory_Izhikevich_somas_genome,
            "E_delay_line":EE_delay_line_genome,
            "I_delay_line":EI_IE_delay_line_genome,
            "EE_process_1":EE_process_1_genome,
            "EE_process_2":EE_process_2_genome,
            "EI_process_1":EI_process_1_genome,
            "EI_process_2":EI_process_2_genome
        },
        "connections":[
            [ 
                ["input", "soma"],
                ["input", "static_input"]
            ],
            [
                ["soma", "EE_delay_line"],
                ["spiked_neurons", "spike_source"]
            ],
            [
                ["soma", "EI_IE_delay_line"],
                ["spiked_neurons", "spike_source"]
            ],
            [
                ["EE_delay_line", "EE_process_1-arborizer"],
                ["spike_output", "spike_source"]
            ],
            [
                ["EE_delay_line", "EE_process_2-arborizer"],
                ["spike_output", "spike_source"]
            ],
            [
                ["EI_IE_delay_line", "EI_process_1-arborizer"],
                ["spike_output", "spike_source"]
            ],
            [
                ["EI_IE_delay_line", "EI_process_2-arborizer"],
                ["spike_output", "spike_source"]
            ],
            
        ]
            
    }

    

    inhibitory_izhikevich_soma_genome = {
        "identifier":"soma",
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
            "inputs": ["E_soma-process_1-dendritic_spine", "E_soma-process_2-dendritic_spine"], # ToDo: there is no error when this mismatches with the names in connections
            "memories":["spiked_neurons"]
            }
        
    }

    neighbours_1 = [(1,-1),(-1,-1),(0,-1),(2,-2)]
    I_axonal_spine_genome = {
        "identifier":"axonal_spine",
        "parameters":{
            "time_step":time_step,
            "type":"DynamicalAxonalTerminalMarkramEtal1998Node",
            "population_size": (population_size[0], population_size[1], len(neighbours_1)),
            "synapse_type":"inhibitory",
            "resting_utilization_of_synaptic_efficacy": {
                "distribution_type": "normal",
                "distribution_parameters": {
                    "loc":0.25,
                    "scale": 0.25/2
                }
            },
            "time_constant_depression":{
                "distribution_type":"normal",
                "distribution_parameters":{
                    "loc":0.7,
                    "scale":0.7/2
                }
            },
            "time_constant_facilitation":{
                "distribution_type":"normal",
                "distribution_parameters":{
                    "loc":0.02,
                    "scale":0.02/2
                }
            },
            "absolute_synaptic_efficacy":{
                "distribution_type":"normal",
                "distribution_parameters": {
                    "loc":-19,
                    "scale":19
                }
            },
            "memories":[]

        }
    }

    IE_process_1_genome = {
        "identifier":"IE_process_1",
        "nodes":{
            "arborizer":arborizer_genome,
            "axonal_spine":I_axonal_spine_genome,
            "dendritic_spine":dendritic_spine_genome
        },
        "connections": [
            [
                ["arborizer", "axonal_spine"],
                ["connection_array", "spike_source"]
            ],
            [
                ["axonal_spine", "dendritic_spine"],
                ["synaptic_response", "synaptic_input"]
            ],
        ]
    }

    IE_process_2_genome = bs.copy_graph_genome(IE_process_1_genome, "IE_process_2")
    
    neighbours_2 = [(1,-2),(0,-2),(-1,-2),(-2,-2)]
    IE_process_2_genome["nodes"]["arborizer"]["parameters"]["connection_relative_position"] = neighbours_2



    inhibitory_neuron_genome = {
        "identifier":"inhibitory_neuron",
        "nodes":{
            "input":input_genome,
            "soma":inhibitory_izhikevich_soma_genome,
            "I_delay_line":EI_IE_delay_line_genome,
            "IE_process_1":IE_process_1_genome,
            "IE_process_2":IE_process_2_genome,
        },
        "connections":[
            [
                ["soma", "EI_IE_delay_line"],
                ["spiked_neurons", "spike_source"]
            ],
            [
                ["soma", "EI_IE_delay_line"],
                ["spiked_neurons", "spike_source"]
            ],
            [
                ["EI_IE_delay_line", "IE_process_1-arborizer"],
                ["spike_output", "spike_source"]
            ],
            [
                ["EI_IE_delay_line", "IE_process_2-arborizer"],
                ["spike_output", "spike_source"]
            ]
        ]
            
    }

    population_genome = {
        "identifier": "E_I_network",
        "nodes":{
            "excitatory_neuron":excitatory_neuron_genome,
            "inhibitory_neuron":inhibitory_neuron_genome
        },
        "connections": [
            [
                ["excitatory_neuron-EE_process_1-dendritic_spine", "excitatory_neuron-soma"],
                ["synaptic_output_summed", "process_1-dendritic_spine"]
            ],
            [
                ["excitatory_neuron-EE_process_2-dendritic_spine", "excitatory_neuron-soma"],
                ["synaptic_output_summed", "process_2-dendritic_spine"]
            ],
            [
                ["excitatory_neuron-EI_process_1-dendritic_spine", "inhibitory_neuron-soma"],
                ["synaptic_output_summed", "E_soma-process_1-dendritic_spine"]
            ],
            [
                ["excitatory_neuron-EI_process_2-dendritic_spine", "inhibitory_neuron-soma"],
                ["synaptic_output_summed", "E_soma-process_2-dendritic_spine"]
            ],
            [
                ["inhibitory_neuron-IE_process_1-dendritic_spine", "excitatory_neuron-soma"],
                ["synaptic_output_summed", "I_soma-process_1-dendritic_spine"]
            ],
            [
                ["inhibitory_neuron-IE_process_2-dendritic_spine", "excitatory_neuron-soma"],
                ["synaptic_output_summed", "I_soma-process_2-dendritic_spine"]
            ]
        ]
    }
    with Client(n_workers = 30) as client:
        graph = bs.DistributedGraph(client, {
            "IzhikevichNode":bs.IzhikevichNode, 
            "DelayLineNode":bs.DelayLineNode, 
            "ArborizerNode":bs.ArborizerNode, 
            "DynamicalAxonalTerminalMarkramEtal1998Node":bs.DynamicalAxonalTerminalMarkramEtal1998Node,
            "DendriticSpineMaasNode":bs.DendriticSpineMaasNode,
            "StaticInput":bs.StaticInput 
            })

        graph.construct_distributed_graph(population_genome)

        t0 = time.time()
        t_last = time.time()
        for t in range(2000):
            #print(t)
            graph.increment()
            if t%100 == 0:
                print("timestep: ",t, "time used on last 100 timesteps: ",time.time()- t_last)
                t_last = time.time()
        print("Total time used: ",time.time() - t0)

        graph.save_memories("test_2d_culture_3_neuron_types")