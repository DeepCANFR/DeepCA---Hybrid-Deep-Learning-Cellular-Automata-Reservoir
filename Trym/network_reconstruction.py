import brainslicer as rm
import matplotlib.pyplot as plt
#import tensorflow as tf
#from scipy import stats
import dask
from dask.distributed import Client, LocalCluster
#from dask.distributed import performance_report
import numpy as ncp
import sys

import numpy as np
import numpy as ncp
import cv2

import cloudpickle
import pickle
import time

if __name__ == '__main__':
    # cluster = SLURMCluster(cores=32,
    #                    processes=4,
    #                    memory="1024GB",
    #                    project="dashproject",
    #                    walltime="01:00:00",
    #                    queue="defq")


    spike_trains_base = ncp.load("spike_trains_base.npy")
    spike_trains_jittered = ncp.load("spike_trains_jittered.npy")
    simulation_length = 2000
    distances = [0, 0.01, 0.02, 0.04]
   
    stop_simulation = False

    for distance_nr in range(4):

        for train_nr in range(10):
            with Client(n_workers = 6) as client:
                print("workers initialized")
                time.sleep(1)

                if distance_nr == 0:
                    stimulation_train = spike_trains_base[train_nr, :]
                else:
                    stimulation_train = spike_trains_jittered[train_nr, :, distance_nr -1]

                #simulation_length = len(stimulation_train)
                #stimulation_train[1000:] = 0
                with open('test_network.pkl', 'rb') as handle:
                    network_dict = cloudpickle.load(handle)
                print("loaded network")

                with open('test_inputs.pkl', 'rb') as handle:
                    input_dict = cloudpickle.load(handle)
                print("loaded input network")
                # homogenous_Circuit_equation_somas_1
                # heterogenous_Izhikevich_1
                with open("heterogenous_Izhikevich_1.pkl", 'rb') as handle:
                    somas = cloudpickle.load(handle)
                print("loaded somas")
                print(__name__)


                time_step = 1
                population_size = (10,100) #image_train[0,:,:].shape
                upper_limit = 10000

                neuron_populations = {}
                # re create the base neurons and all their components
                for key in network_dict:
                    neuron_populations[key] = rm.NeuronsFullyDistributed(client)
                    neuron_populations[key].reconstruct_neuron(network_dict[key])
                # ensure that every neuron has access to the the other neurons
                for key in neuron_populations:
                    neuron_populations[key].set_connected_neurons(neuron_populations)
                for key in network_dict:
                    #important that this is done before set_somas since it would overwrite it
                    neuron_populations[key].reconstruct_component_states(network_dict[key])
                for key in network_dict:
                    soma = somas[key]
                    neuron_populations[key].set_soma(soma)

                # set up the connecteions between each component
                for key in network_dict:
                    neuron_populations[key].reconstruct_connections()


                input_populations = {}
                for key in input_dict:
                    input_populations[key] = rm.InputNeurons(client)
                    input_populations[key].reconstruct_neuron(input_dict[key])
                for key in input_populations:
                    input_populations[key].set_connected_neurons(neuron_populations)
                # set up the connecteions between each component
                for key in input_dict:
                    input_populations[key].reconstruct_component_states(input_dict[key])
                for key in input_dict:
                    input_populations[key].reconstruct_connections()


                image = ncp.zeros((int(population_size[0]*3), population_size[1], 3))



                nr_of_layers = 6
                network_history = ncp.zeros((population_size[0], population_size[1], nr_of_layers, simulation_length))
                stop_simulation = False

                setling_time = 1 #ms
                setling_time_steps = int(setling_time/time_step)


                for t in range(simulation_length):
                    # input spikes are generated
                    print(t)
                    #print("t: ", ncp.round(t*time_step,2), " Spike: ", stimulation_train[t], end = '\r')#, ncp.amax(inputs.interfacable))

                    for ID in neuron_populations:
                        neuron_populations[ID].update_current_values()
                    for ID in input_populations:
                        input_populations[ID].update_current_values()


                    for ID in neuron_populations:
                        neuron_populations[ID].get_results()
                    for ID in input_populations:
                        input_populations[ID].get_results()

                    for ID in neuron_populations:
                        neuron_populations[ID].compute_new_values()
                    for ID in input_populations:
                        if t < len(stimulation_train):
                            input_populations[ID].compute_new_values(stimulation_train[t])
                        else:
                            input_populations[ID].compute_new_values(0)

                    #time.sleep(2)
                    for ID in neuron_populations:
                        neuron_populations[ID].get_results()
                    for ID in input_populations:
                        input_populations[ID].get_results()

                    '''
                    image_E1 = input_populations["E1_input"].soma.interfacable
                    image_E2 = input_populations["E2_input"].soma.interfacable
                    image_E3 = input_populations["E3_input"].soma.interfacable
                    image_I1 = input_populations["I1_input"].soma.interfacable
                    image_I2 = input_populations["I2_input"].soma.interfacable
                    image_I3 = input_populations["I3_input"].soma.interfacable
                    '''
                    image_E1 = neuron_populations["E1_soma"].soma.interfacable
                    image_E2 = neuron_populations["E2_soma"].soma.interfacable
                    image_E3 = neuron_populations["E3_soma"].soma.interfacable
                    #image_I1 = input_populations["E1_input"].soma.interfacable
                    #image_I2 = input_populations["E2_input"].soma.interfacable
                    #image_I3 = input_populations["E3_input"].soma.interfacable
                    image_I1 = neuron_populations["I1_soma"].soma.interfacable
                    image_I2 = neuron_populations["I2_soma"].soma.interfacable
                    image_I3 = neuron_populations["I3_soma"].soma.interfacable
                    #print(ncp.amax(image_E1))
                    network_history[:,:,0,t] = image_E1
                    network_history[:,:,1,t] = image_E2
                    network_history[:,:,2,t] = image_E3
                    network_history[:,:,3,t] = image_I1
                    network_history[:,:,4,t] = image_I2
                    network_history[:,:,5,t] = image_I3


                    E_image = ncp.concatenate((image_E1, image_E2, image_E3), axis = 0)
                    I_image = ncp.concatenate((image_I1, image_I2, image_I3), axis = 0)

                    image[:,:,2] = E_image
                    image[:,:,0] = I_image
                    image*=255
                    #image = ncp.concatenate((image_E1, image_E2, image_E3, image_I1, image_I2, image_I3), axis = 0)
                    #image = inputs.interfacable
                    #print(ncp.amax(image[:,:,1]))
                    image_shape = ncp.array(image.shape[0:2])
                    image_shape *= 15
                    image_shape = image_shape[::-1]
                    image_shape = tuple(image_shape)

                    #image = ncp.asnumpy(image)
                    image_out = cv2.resize(image,image_shape)

                    '''
                    voltages = E1_neurons.components["E1_soma"].current_somatic_voltages
                    timeline.append((t+1 )*time_step)
                    v.append(voltages[5,10])
                    u.append(E1_neurons.components["E1_soma"].current_u[5,10])
                    spikes_plot.append(image_E1[5,10])
                    '''

                    #print(ncp.amax(image))
                    image_out = image_out.astype(ncp.uint8)
                    cv2.imshow('frame', image_out)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_simulation = True
                        break
                # network_history_base_1_homogenous_circuit_equation
                # network_history_jittered_1_homogenous_circuit_equation

                # network_history_test_1
                # network_history_jittered_1
                print("attempting to save ", stop_simulation)
                if stop_simulation == False:
                    print("saving")
                    filename = str(distances[distance_nr]) + "_train_nr_" + str(train_nr) +  "_network_history_1_heterogenous_Izhikevich_long_trials_somas_1_distance_by_jitter"
                    print(filename)
                    ncp.save(filename, network_history)
                if stop_simulation == True:
                    for ID in neuron_populations:
                        neuron_populations[ID].get_results()

                print("Shutting down client")
                client.shutdown()
                
                time.sleep(2)

            #cv2.destroyAllWindows()
            '''
            timeline = ncp.array(timeline)
            plt.plot(timeline, ncp.array(v), label = "somatic v")
            plt.plot(timeline, ncp.array(u), label = "recovery u")
            plt.plot(timeline, ncp.array(spikes_plot), label = "spikes")
            plt.legend()
            plt.show()
            #print(v)
            #print(u)
            '''

            if stop_simulation:
                break
        if stop_simulation:
            break



    cv2.destroyAllWindows()
