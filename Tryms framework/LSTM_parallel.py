# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:14:08 2021

@author: trymlind
"""
if __name__ == '__main__':

    import parallel_realistic_module as rm
    import matplotlib.pyplot as plt
    #import tensorflow as tf
    from scipy import stats
    import dask
    from dask.distributed import Client, LocalCluster
    import sys


    import numpy as ncp
    import cv2

    '''
    Load stimuli
    #######################################################################################
    '''
    #(image_train, label_train), (image_test, label_test) = tf.keras.datasets.mnist.load_data()


    '''
    Set parameters
    '''
    with Client(n_workers = 6) as client:
        weight_scaling = 1

        time_step = 0.1
        population_size = (50,50) #image_train[0,:,:].shape
        upper_limit = 10000

        excitatory_soma_parameters = {}
        excitatory_soma_parameters["population_size"] = population_size
        excitatory_soma_parameters["membrane_time_constant"] = 15 # ms
        excitatory_soma_parameters["absolute_refractory_period"] = 3 # ms
        excitatory_soma_parameters["threshold"] = 19 # mv
        excitatory_soma_parameters["reset_voltage"] = 13.5 # mv
        excitatory_soma_parameters["background_current"] = 13.5 # nA
        excitatory_soma_parameters["input_resistance"] = 1 # M_Ohm
        excitatory_soma_parameters["refractory_period"] = 3
        excitatory_soma_parameters["time_step"] = time_step
        excitatory_soma_parameters["temporal_upper_limit"] = upper_limit

        inhibitory_soma_parameters = {}
        inhibitory_soma_parameters["population_size"] = population_size
        inhibitory_soma_parameters["membrane_time_constant"] = 30 # ms
        inhibitory_soma_parameters["absolute_refractory_period"] = 2 # ms
        inhibitory_soma_parameters["threshold"] = 15 # mv
        inhibitory_soma_parameters["reset_voltage"] = 13.5 # mv
        inhibitory_soma_parameters["background_current"] = 13.5 # nA
        inhibitory_soma_parameters["input_resistance"] = 1 # M_Ohm
        inhibitory_soma_parameters["refractory_period"] = 2
        inhibitory_soma_parameters["time_step"] = time_step
        inhibitory_soma_parameters["temporal_upper_limit"] = upper_limit

        # to do: make all paramets only have positive values
        excitatory_to_excitatory_dynamical_synapse_parameters = {}
        excitatory_to_excitatory_dynamical_synapse_parameters["resting_utilization_of_synaptic_efficacy"] = {"distribution":"normal", "mean":0.5, "SD":0.5/2}# U
        excitatory_to_excitatory_dynamical_synapse_parameters["time_constant_depresssion"] = {"distribution":"normal", "mean":1.1, "SD":1.1/2}# in Maas et al: D, in Markram et al: tau_depression  expressed in seconds
        excitatory_to_excitatory_dynamical_synapse_parameters["time_constant_facilitation"] = {"distribution":"normal", "mean":0.5, "SD":0.5/2}# in Maas et al: F, in Markram et al: tau_facil
        # to do: this should be from a gamma distribution I think, but I don't understand how they made it
        excitatory_to_excitatory_dynamical_synapse_parameters["absolute_synaptic_efficacy"] = {"distribution":"normal", "mean":30*weight_scaling, "SD":30*weight_scaling}# in Maas et al: A, in Markram et al: A
        excitatory_to_excitatory_dynamical_synapse_parameters["type"] = "excitatory"
        excitatory_to_excitatory_dynamical_synapse_parameters["time_step"] = time_step
        excitatory_to_excitatory_dynamical_synapse_parameters["temporal_upper_limit"] = upper_limit

        excitatory_to_inhibitory_dynamical_synapse_parameters = {}
        excitatory_to_inhibitory_dynamical_synapse_parameters["resting_utilization_of_synaptic_efficacy"] = {"distribution":"normal", "mean":0.5, "SD":0.5/2}# U  Strange, the setting from the paper is 0.05, but this results in the inhibitory neurons not firing
        excitatory_to_inhibitory_dynamical_synapse_parameters["time_constant_depresssion"] = {"distribution":"normal", "mean":0.125, "SD":0.125/2} # in Maas et al: D, in Markram et al: tau_depression  expressed in seconds
        excitatory_to_inhibitory_dynamical_synapse_parameters["time_constant_facilitation"] = {"distribution":"normal", "mean":1.2, "SD":1.2/2} # in Maas et al: F, in Markram et al: tau_facil
        # to do: this should be from a gamma distribution I think, but I don't understand how they made it
        excitatory_to_inhibitory_dynamical_synapse_parameters["absolute_synaptic_efficacy"] = {"distribution":"normal", "mean":60*weight_scaling, "SD":60*weight_scaling}# in Maas et al: A, in Markram et al: A
        excitatory_to_inhibitory_dynamical_synapse_parameters["type"] = "excitatory"
        excitatory_to_inhibitory_dynamical_synapse_parameters["time_step"] = time_step
        excitatory_to_inhibitory_dynamical_synapse_parameters["temporal_upper_limit"] = upper_limit

        inhibitory_to_excitatory_dynamical_synapse_parameters = {}
        inhibitory_to_excitatory_dynamical_synapse_parameters["resting_utilization_of_synaptic_efficacy"] = {"distribution":"normal", "mean":0.25, "SD":0.25/2} # U
        inhibitory_to_excitatory_dynamical_synapse_parameters["time_constant_depresssion"] = {"distribution":"normal", "mean":0.7, "SD":0.7/2} # in Maas et al: D, in Markram et al: tau_depression  expressed in seconds
        inhibitory_to_excitatory_dynamical_synapse_parameters["time_constant_facilitation"] = {"distribution":"normal", "mean":0.02, "SD":0.02/2} # in Maas et al: F, in Markram et al: tau_facil
        # to do: this should be from a gamma distribution I think, but I don't understand how they made it
        inhibitory_to_excitatory_dynamical_synapse_parameters["absolute_synaptic_efficacy"] = {"distribution":"normal", "mean":-19*weight_scaling, "SD":19*weight_scaling}# in Maas et al: A, in Markram et al: A
        inhibitory_to_excitatory_dynamical_synapse_parameters["type"] = "inhibitory"
        inhibitory_to_excitatory_dynamical_synapse_parameters["time_step"] = time_step
        inhibitory_to_excitatory_dynamical_synapse_parameters["temporal_upper_limit"] = upper_limit

        inhibitory_to_inhibitory_dynamical_synapse_parameters = {}
        inhibitory_to_inhibitory_dynamical_synapse_parameters["resting_utilization_of_synaptic_efficacy"] = {"distribution":"normal", "mean":0.32, "SD":0.32/2} # U
        inhibitory_to_inhibitory_dynamical_synapse_parameters["time_constant_depresssion"] = {"distribution":"normal", "mean":0.144, "SD":0.144/2} # in Maas et al: D, in Markram et al: tau_depression  expressed in seconds
        inhibitory_to_inhibitory_dynamical_synapse_parameters["time_constant_facilitation"] = {"distribution":"normal", "mean":0.06, "SD":0.06/2} # in Maas et al: F, in Markram et al: tau_facil
        # to do: this should be from a gamma distribution I think, but I don't understand how they made it
        inhibitory_to_inhibitory_dynamical_synapse_parameters["absolute_synaptic_efficacy"] = {"distribution":"normal", "mean":-19*weight_scaling, "SD":19*weight_scaling}# in Maas et al: A, in Markram et al: A
        inhibitory_to_inhibitory_dynamical_synapse_parameters["type"] = "inhibitory"
        inhibitory_to_inhibitory_dynamical_synapse_parameters["time_step"] = time_step
        inhibitory_to_inhibitory_dynamical_synapse_parameters["temporal_upper_limit"] = upper_limit


        EE_dendritic_arbor_parameters = {}
        projection_template = ncp.ones((3,3))
        #projection_template[8,4:] = 1
        #projection_template[2,0:2] = 1
        #projection_template[1:-1,:] = 0
        #projection_template[2,1] = 1
        #projection_template[2,0:3] = 1
        EE_dendritic_arbor_parameters["projection_template"] = projection_template
        EE_dendritic_arbor_parameters["time_step"] = time_step
        EE_dendritic_arbor_parameters["distance_based_connection_probability"] = {"C":0.3, "lambda_parameter":5}
        EE_dendritic_arbor_parameters["temporal_upper_limit"] = upper_limit
        EE_dendritic_arbor_parameters["boundry_conditions"] = "closed"

        EE_delay_line_parameters = {}
        EE_delay_line_parameters["delay"] = 1.5 # ms
        EE_delay_line_parameters["time_step"] = time_step #ms
        EE_delay_line_parameters["temporal_upper_limit"] = upper_limit

        EI_dendritic_arbor_parameters = {}
        EI_dendritic_arbor_parameters["projection_template"] = ncp.ones((7,7))
        EI_dendritic_arbor_parameters["projection_template"][1:5,1:5] = 0
        EI_dendritic_arbor_parameters["time_step"] = time_step
        EI_dendritic_arbor_parameters["distance_based_connection_probability"] = {"C":0.2, "lambda_parameter":5}
        EI_dendritic_arbor_parameters["temporal_upper_limit"] = upper_limit

        EI_delay_line_parameters = {}
        EI_delay_line_parameters["delay"] = 0.8
        EI_delay_line_parameters["time_step"] = time_step
        EI_delay_line_parameters["temporal_upper_limit"] = upper_limit

        IE_dendritic_arbor_parameters = {}
        IE_dendritic_arbor_parameters["projection_template"] = ncp.ones((5,5))
        IE_dendritic_arbor_parameters["time_step"] = time_step
        IE_dendritic_arbor_parameters["distance_based_connection_probability"] = {"C":0.4, "lambda_parameter":5}
        IE_dendritic_arbor_parameters["temporal_upper_limit"] = upper_limit

        #excitatory dendritic spine
        E_dendritic_Spine_parameters = {}
        E_dendritic_Spine_parameters["time_step"] = time_step
        E_dendritic_Spine_parameters["time_constant"] = 3 # ms
        E_dendritic_Spine_parameters["temporal_upper_limit"] = upper_limit

        #inhibitory dendritic spine
        I_dendritic_Spine_parameters = {}
        I_dendritic_Spine_parameters["time_step"] = time_step
        I_dendritic_Spine_parameters["time_constant"] = 6 # ms
        I_dendritic_Spine_parameters["temporal_upper_limit"] = upper_limit

        IE_delay_line_parameters = {}
        IE_delay_line_parameters["delay"] = 0.8
        IE_delay_line_parameters["time_step"] = time_step
        IE_delay_line_parameters["temporal_upper_limit"] = upper_limit



        P_delta_readout_parameters = {}
        P_delta_readout_parameters["nr_of_readout_neurons"] = 51
        P_delta_readout_parameters["error_tolerance"] = 0.05
        P_delta_readout_parameters["rho"] = 1 # squashing function boundries
        P_delta_readout_parameters["margin"] = 0.02
        P_delta_readout_parameters["clear_margins_importance"] = 1
        P_delta_readout_parameters["learning_rate"] = 0.0025
        P_delta_readout_parameters["temporal_upper_limit"] = upper_limit


        #client = Client()
        '''
        Build neuron
        ######################################################################################
        '''

        future_E_Soma = client.submit(rm.Simple_Integrate_and_Fire_Soma, excitatory_soma_parameters, actors = True)
        future_EE_delay_line = client.submit(rm.Delay_Line, EE_delay_line_parameters, actors = True)
        future_EE_dendritic_arbor = client.submit(rm.Dendritic_Arbor, EE_dendritic_arbor_parameters, actors = True)
        future_EE_axonal_terminal = client.submit(rm.Dynamical_Axonal_Terminal_Markram_etal_1998, excitatory_to_excitatory_dynamical_synapse_parameters, actors = True)
        future_EE_dendritic_spine = client.submit(rm.Dendritic_Spine_Maas, E_dendritic_Spine_parameters, actors = True)
        future_inputs = client.submit(rm.Input_Class, population_size, actors = True)

        E_somas = future_E_Soma.result()
        EE_delay_line = future_EE_delay_line.result()
        EE_dendritic_arbor = future_EE_dendritic_arbor.result()
        EE_axonal_terminal = future_EE_axonal_terminal.result()
        EE_dendritic_spine = future_EE_dendritic_spine.result()
        inputs = future_inputs.result()

        # Note that because the components are in large part built when they are interface
        # You need to get the results after interfacing each component
        future = E_somas.interface(inputs)
        future.result()
        future = EE_delay_line.interface_read_variable(E_somas)
        future.result()
        future = EE_dendritic_arbor.interface(EE_delay_line)
        future.result()
        future = EE_axonal_terminal.interface_read_variable(EE_dendritic_arbor)
        future.result()
        future = EE_dendritic_spine.interface_read_variable(EE_axonal_terminal)
        future.result()
        future = E_somas.interface(EE_dendritic_spine)
        future.result()


        print("killing connections")
        print(ncp.sum(EE_dendritic_arbor.kill_mask))
        future = EE_dendritic_arbor.set_boundry_conditions()
        future.result()
        print(ncp.sum(EE_dendritic_arbor.kill_mask))
        #print(EE_dendritic_arbor.kill_mask)
        print("Killing connections based on distance")
        future = EE_dendritic_arbor.kill_connections_based_on_distance()
        print(future.result())
        #print(ncp.sum(EE_dendritic_arbor.kill_mask))
        #print(EE_dendritic_arbor.kill_mask)


        components = [E_somas, EE_delay_line, EE_dendritic_arbor, EE_axonal_terminal, EE_dendritic_spine]

        simulation_length = 10000
        print("Creating input object")
        #new_input = ncp.random.uniform(0,1,population_size) > 0.7
        new_input = ncp.zeros(population_size)
        new_input[15,15] = 1
        new_input = new_input * 70
        #new_input[-5,-5] = 70
        future = inputs.update_input(new_input)

        future.result()
        print("Input objected created")

        v = []
        s_inputs = []
        s_voltages = []
        time = []


        for t in range(simulation_length):
            time.append(t*time_step)
            print(t*time_step)
            futures = []

            for component in components:
                future = component.update_current_values()
                futures.append(future)
            for future in futures:
                future.result()

            futures = []
            for component in components:
                future = component.compute_new_values()
                futures.append(future)
            for future in futures:
                future.result()
                #print(future.result())

            image = E_somas.interfacable
            voltages = EE_dendritic_spine.interfacable
            v.append(voltages[14,15,:])
            s_inputs.append(E_somas.summed_inputs[14,15])
            s_voltages.append(E_somas.current_somatic_voltages[14,15])
            image_shape = ncp.array(image.shape)
            image_shape *= 10
            image_shape = image_shape[::-1]
            image_shape = tuple(image_shape)
            image = cv2.resize(image,image_shape)
            max = ncp.amax(image)
            print(max)
            image *= 255
            #image /= max
            #image*= 255

            #print(ncp.amax(image))
            image = image.astype(ncp.uint8)
            cv2.imshow('frame', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_simulation = True
                break
        cv2.destroyAllWindows()

        v= ncp.array(v)
        time = ncp.array(time)
        plt.plot(time,v)
        plt.figure()
        s_inputs = ncp.array(s_inputs)
        s_voltages = ncp.array(s_voltages)
        plt.plot(time,s_inputs)
        plt.plot(time,s_voltages)
        plt.show()
