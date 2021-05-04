# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:14:08 2021

@author: trymlind
"""
import parallel_realistic_module as rm
import matplotlib.pyplot as plt
#import tensorflow as tf
from scipy import stats
import dask
from dask.distributed import Client, LocalCluster
from dask.distributed import performance_report

import sys

import numpy as np
import numpy as ncp
import cv2

if __name__ == '__main__':

    with Client(n_workers = 6) as client:

        '''
        Load stimuli
        #######################################################################################
        '''
        #(image_train, label_train), (image_test, label_test) = tf.keras.datasets.mnist.load_data()


        '''
        Set parameters
        ######################################################################################
        The first step is to setup the genereic parameters dictionaries. Since we are building
        three layers (x2 one for inhibitor and one for excitatory) we will have many connections
        That utilize the same type components with the same parameter settings. Since we are
        using them for multiple components we do not give them an ID at this point other than
        the object name which specifies which type of component the parameters are for
        '''

        weight_scaling = 1

        time_step = 0.5
        population_size = (10,100) #image_train[0,:,:].shape
        upper_limit = 10000

        '''
        Somas
        '''
        # Excitatory (E)
        E_soma_parameters = {}
        E_soma_parameters["population_size"] = population_size
        E_soma_parameters["membrane_time_constant"] = 15 # ms
        E_soma_parameters["absolute_refractory_period"] = 3 # ms
        E_soma_parameters["threshold"] = 19 # mv
        E_soma_parameters["reset_voltage"] = 13.5 # mv
        E_soma_parameters["background_current"] = 13.5 # nA
        E_soma_parameters["input_resistance"] = 1 # M_Ohm
        E_soma_parameters["refractory_period"] = 3
        E_soma_parameters["time_step"] = time_step
        E_soma_parameters["temporal_upper_limit"] = upper_limit

        # Inhibitory (I)
        I_soma_parameters = {}
        I_soma_parameters["population_size"] = population_size
        I_soma_parameters["membrane_time_constant"] = 30 # ms
        I_soma_parameters["absolute_refractory_period"] = 2 # ms
        I_soma_parameters["threshold"] = 15 # mv
        I_soma_parameters["reset_voltage"] = 13.5 # mv
        I_soma_parameters["background_current"] = 13.5 # nA
        I_soma_parameters["input_resistance"] = 1 # M_Ohm
        I_soma_parameters["refractory_period"] = 2
        I_soma_parameters["time_step"] = time_step
        I_soma_parameters["temporal_upper_limit"] = upper_limit

        '''
        Dynamical synapses
        '''
        # Excitatory to Excitatory (EE)
        EE_dynamical_synapse_parameters = {}
        EE_dynamical_synapse_parameters["resting_utilization_of_synaptic_efficacy"] = {"distribution":"normal", "mean":0.5, "SD":0.5/2}# U
        EE_dynamical_synapse_parameters["time_constant_depresssion"] = {"distribution":"normal", "mean":1.1, "SD":1.1/2}# in Maas et al: D, in Markram et al: tau_depression  expressed in seconds
        EE_dynamical_synapse_parameters["time_constant_facilitation"] = {"distribution":"normal", "mean":0.5, "SD":0.5/2}# in Maas et al: F, in Markram et al: tau_facil
        # to do: this should be from a gamma distribution I think, but I don't understand how they made it
        EE_dynamical_synapse_parameters["absolute_synaptic_efficacy"] = {"distribution":"normal", "mean":30*weight_scaling, "SD":30*weight_scaling}# in Maas et al: A, in Markram et al: A
        EE_dynamical_synapse_parameters["type"] = "excitatory"
        EE_dynamical_synapse_parameters["time_step"] = time_step
        EE_dynamical_synapse_parameters["temporal_upper_limit"] = upper_limit

        #Excitatory to Inhibitory (EI)
        EI_dynamical_synapse_parameters = {}
        EI_dynamical_synapse_parameters["resting_utilization_of_synaptic_efficacy"] = {"distribution":"normal", "mean":0.5, "SD":0.5/2}# U  Strange, the setting from the paper is 0.05, but this results in the inhibitory neurons not firing
        EI_dynamical_synapse_parameters["time_constant_depresssion"] = {"distribution":"normal", "mean":0.125, "SD":0.125/2} # in Maas et al: D, in Markram et al: tau_depression  expressed in seconds
        EI_dynamical_synapse_parameters["time_constant_facilitation"] = {"distribution":"normal", "mean":1.2, "SD":1.2/2} # in Maas et al: F, in Markram et al: tau_facil
        # to do: this should be from a gamma distribution I think, but I don't understand how they made it
        EI_dynamical_synapse_parameters["absolute_synaptic_efficacy"] = {"distribution":"normal", "mean":60*weight_scaling, "SD":60*weight_scaling}# in Maas et al: A, in Markram et al: A
        EI_dynamical_synapse_parameters["type"] = "excitatory"
        EI_dynamical_synapse_parameters["time_step"] = time_step
        EI_dynamical_synapse_parameters["temporal_upper_limit"] = upper_limit

        # Inhibitory to Excitatory (IE)
        IE_dynamical_synapse_parameters = {}
        IE_dynamical_synapse_parameters["resting_utilization_of_synaptic_efficacy"] = {"distribution":"normal", "mean":0.25, "SD":0.25/2} # U
        IE_dynamical_synapse_parameters["time_constant_depresssion"] = {"distribution":"normal", "mean":0.7, "SD":0.7/2} # in Maas et al: D, in Markram et al: tau_depression  expressed in seconds
        IE_dynamical_synapse_parameters["time_constant_facilitation"] = {"distribution":"normal", "mean":0.02, "SD":0.02/2} # in Maas et al: F, in Markram et al: tau_facil
        # to do: this should be from a gamma distribution I think, but I don't understand how they made it
        IE_dynamical_synapse_parameters["absolute_synaptic_efficacy"] = {"distribution":"normal", "mean":-19*weight_scaling, "SD":19*weight_scaling}# in Maas et al: A, in Markram et al: A
        IE_dynamical_synapse_parameters["type"] = "inhibitory"
        IE_dynamical_synapse_parameters["time_step"] = time_step
        IE_dynamical_synapse_parameters["temporal_upper_limit"] = upper_limit

        # Inhibitory to Inhibitory (II)
        II_dynamical_synapse_parameters = {}
        II_dynamical_synapse_parameters["resting_utilization_of_synaptic_efficacy"] = {"distribution":"normal", "mean":0.32, "SD":0.32/2} # U
        II_dynamical_synapse_parameters["time_constant_depresssion"] = {"distribution":"normal", "mean":0.144, "SD":0.144/2} # in Maas et al: D, in Markram et al: tau_depression  expressed in seconds
        II_dynamical_synapse_parameters["time_constant_facilitation"] = {"distribution":"normal", "mean":0.06, "SD":0.06/2} # in Maas et al: F, in Markram et al: tau_facil
        # to do: this should be from a gamma distribution I think, but I don't understand how they made it
        II_dynamical_synapse_parameters["absolute_synaptic_efficacy"] = {"distribution":"normal", "mean":-19*weight_scaling, "SD":19*weight_scaling}# in Maas et al: A, in Markram et al: A
        II_dynamical_synapse_parameters["type"] = "inhibitory"
        II_dynamical_synapse_parameters["time_step"] = time_step
        II_dynamical_synapse_parameters["temporal_upper_limit"] = upper_limit



        '''
        Delay lines (axons)
        '''

        EE_delay_line_parameters = {}
        EE_delay_line_parameters["delay"] = 2#1.5 # ms
        EE_delay_line_parameters["time_step"] = time_step #ms
        EE_delay_line_parameters["temporal_upper_limit"] = upper_limit

        EI_delay_line_parameters = {}
        EI_delay_line_parameters["delay"] = 1#0.8
        EI_delay_line_parameters["time_step"] = time_step
        EI_delay_line_parameters["temporal_upper_limit"] = upper_limit

        IE_delay_line_parameters = {}
        IE_delay_line_parameters["delay"] = 1#0.8
        IE_delay_line_parameters["time_step"] = time_step
        IE_delay_line_parameters["temporal_upper_limit"] = upper_limit

        II_delay_line_parameters = {}
        II_delay_line_parameters["delay"] = 1#0.8
        II_delay_line_parameters["time_step"] = time_step
        II_delay_line_parameters["temporal_upper_limit"] = upper_limit

        '''
        arbors
        '''
        EE_dendritic_arbor_parameters = {}
        projection_template = ncp.ones((5,5))
        #projection_template[8,4:] = 1
        #projection_template[2,0:2] = 1
        #projection_template[1:-1,:] = 0
        #projection_template[1,0] = 1
        EE_dendritic_arbor_parameters["projection_template"] = projection_template
        EE_dendritic_arbor_parameters["time_step"] = time_step
        EE_dendritic_arbor_parameters["distance_based_connection_probability"] = {"C":0.3, "lambda_parameter":5}
        EE_dendritic_arbor_parameters["temporal_upper_limit"] = upper_limit
        EE_dendritic_arbor_parameters["boundry_conditions"] = "closed"

        EI_dendritic_arbor_parameters = {}
        EI_dendritic_arbor_parameters["projection_template"] = ncp.ones((7,7))
        #EI_dendritic_arbor_parameters["projection_template"][1:5,1:5] = 0
        EI_dendritic_arbor_parameters["time_step"] = time_step
        EI_dendritic_arbor_parameters["distance_based_connection_probability"] = {"C":0.2, "lambda_parameter":5}
        EI_dendritic_arbor_parameters["temporal_upper_limit"] = upper_limit
        EI_dendritic_arbor_parameters["boundry_conditions"] = "closed"

        IE_dendritic_arbor_parameters = {}
        IE_dendritic_arbor_parameters["projection_template"] = ncp.ones((5,5))
        IE_dendritic_arbor_parameters["time_step"] = time_step
        IE_dendritic_arbor_parameters["distance_based_connection_probability"] = {"C":0.4, "lambda_parameter":5}
        IE_dendritic_arbor_parameters["temporal_upper_limit"] = upper_limit
        IE_dendritic_arbor_parameters["boundry_conditions"] = "closed"

        II_dendritic_arbor_parameters = {}
        II_dendritic_arbor_parameters["projection_template"] = ncp.ones((5,5))
        II_dendritic_arbor_parameters["time_step"] = time_step
        II_dendritic_arbor_parameters["distance_based_connection_probability"] = {"C":0.4, "lambda_parameter":5}
        II_dendritic_arbor_parameters["temporal_upper_limit"] = upper_limit
        II_dendritic_arbor_parameters["boundry_conditions"] = "closed"

        '''
        Dendritic spines
        '''
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


        '''
        Here we create the dictionaries that contain the parameter dicts for the sequence
        of component a connection consist of. Note that we use the unique_ID_dict_creator object
        to create a copy of the generic dict for a component that has a unique ID attached to it.
        This ID is used when the connnection is built later.
        '''
        # use this class to create unique ID's for every parameter dict
        unique_ID_dict_creator = rm.Unique_ID_Dict_Creator(30)

        # Setup connection parameter dicts
        # Excitatory self connections
        self_E_1_parameter_dict = {}
        self_E_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
        self_E_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
        self_E_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
        self_E_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

        self_E_2_parameter_dict = {}
        self_E_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
        self_E_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
        self_E_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
        self_E_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

        self_E_3_parameter_dict = {}
        self_E_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
        self_E_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
        self_E_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
        self_E_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

        # Inhibitory self connections
        self_I_1_parameter_dict = {}
        self_I_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
        self_I_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
        self_I_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
        self_I_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

        self_I_2_parameter_dict = {}
        self_I_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
        self_I_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
        self_I_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
        self_I_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

        self_I_3_parameter_dict = {}
        self_I_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
        self_I_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
        self_I_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
        self_I_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

        # Excitatory to Excitatory connection from layer x to layer y
        EE_1_2_parameter_dict = {}
        EE_1_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
        EE_1_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
        EE_1_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
        EE_1_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

        EE_1_3_parameter_dict = {}
        EE_1_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
        EE_1_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
        EE_1_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
        EE_1_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

        EE_2_1_parameter_dict = {}
        EE_2_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
        EE_2_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
        EE_2_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
        EE_2_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

        EE_2_3_parameter_dict = {}
        EE_2_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
        EE_2_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
        EE_2_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
        EE_2_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

        # Excitatory to Excitatory connection from layer 3 to layer x
        EE_3_1_parameter_dict = {}
        EE_3_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
        EE_3_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
        EE_3_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
        EE_3_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

        EE_3_2_parameter_dict = {}
        EE_3_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
        EE_3_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
        EE_3_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
        EE_3_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)




        # Excitatory to Excitatory connection from layer 1 to layer 2
        II_1_2_parameter_dict = {}
        II_1_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
        II_1_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
        II_1_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
        II_1_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

        # Excitatory to Excitatory connection from layer 1 to layer 3
        II_1_3_parameter_dict = {}
        II_1_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
        II_1_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
        II_1_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
        II_1_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

        # Excitatory to Excitatory connection from layer 2 to layer 1
        II_2_1_parameter_dict = {}
        II_2_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
        II_2_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
        II_2_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
        II_2_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

        # Excitatory to Excitatory connection from layer 2 to layer 3
        II_2_3_parameter_dict = {}
        II_2_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
        II_2_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
        II_2_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
        II_2_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

        # Excitatory to Excitatory connection from layer 3 to layer x
        II_3_1_parameter_dict = {}
        II_3_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
        II_3_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
        II_3_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
        II_3_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

        II_3_2_parameter_dict = {}
        II_3_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
        II_3_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
        II_3_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
        II_3_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)


        # Excitatory to Inhibitory connections
        EI_1_1_parameter_dict = {}
        EI_1_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
        EI_1_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
        EI_1_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
        EI_1_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

        EI_1_2_parameter_dict = {}
        EI_1_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
        EI_1_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
        EI_1_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
        EI_1_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

        EI_1_3_parameter_dict = {}
        EI_1_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
        EI_1_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
        EI_1_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
        EI_1_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

        ##############
        EI_2_1_parameter_dict = {}
        EI_2_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
        EI_2_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
        EI_2_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
        EI_2_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

        EI_2_2_parameter_dict = {}
        EI_2_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
        EI_2_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
        EI_2_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
        EI_2_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

        EI_2_3_parameter_dict = {}
        EI_2_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
        EI_2_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
        EI_2_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
        EI_2_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

        #############
        EI_3_1_parameter_dict = {}
        EI_3_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
        EI_3_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
        EI_3_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
        EI_3_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

        EI_3_2_parameter_dict = {}
        EI_3_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
        EI_3_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
        EI_3_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
        EI_3_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

        EI_3_3_parameter_dict = {}
        EI_3_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
        EI_3_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
        EI_3_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
        EI_3_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

        # Inhibitory to Excitatory connections
        IE_1_1_parameter_dict = {}
        IE_1_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
        IE_1_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
        IE_1_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
        IE_1_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

        IE_1_2_parameter_dict = {}
        IE_1_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
        IE_1_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
        IE_1_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
        IE_1_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

        IE_1_3_parameter_dict = {}
        IE_1_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
        IE_1_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
        IE_1_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
        IE_1_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

        ############
        IE_2_1_parameter_dict = {}
        IE_2_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
        IE_2_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
        IE_2_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
        IE_2_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

        IE_2_2_parameter_dict = {}
        IE_2_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
        IE_2_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
        IE_2_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
        IE_2_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

        IE_2_3_parameter_dict = {}
        IE_2_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
        IE_2_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
        IE_2_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
        IE_2_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

        ##########
        IE_3_1_parameter_dict = {}
        IE_3_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
        IE_3_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
        IE_3_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
        IE_3_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

        IE_3_2_parameter_dict = {}
        IE_3_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
        IE_3_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
        IE_3_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
        IE_3_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

        IE_3_3_parameter_dict = {}
        IE_3_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
        IE_3_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
        IE_3_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
        IE_3_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)




        '''
        Build neuron
        ######################################################################################
        Here we instantiate each neuron population. The first step of this process is to create
        Neurons objects. Upon initiation these will contain future somas and not much else.
        We can then use the connection dictionaries to create connections between populations

        '''
        #client = Client(asynchronous = True, n_workers = 4, threads_per_worker = 2)

        inputs = client.submit(rm.Input_Class, population_size, actors = True)
        inputs = inputs.result()

        # Here we first create neuron object. Initioally these only contain the soma of the neurons
        # as they are not connected to anything and thus don't have dendrites or axons
        E1_neurons = rm.Neurons(rm.Circuit_Equation_Integrate_and_Fire_Soma, E_soma_parameters, 0, "E1_soma", client)
        E2_neurons = rm.Neurons(rm.Circuit_Equation_Integrate_and_Fire_Soma, E_soma_parameters, 1, "E2_soma", client)
        E3_neurons = rm.Neurons(rm.Circuit_Equation_Integrate_and_Fire_Soma, E_soma_parameters, 2, "E3_soma", client)

        I1_neurons = rm.Neurons(rm.Circuit_Equation_Integrate_and_Fire_Soma, I_soma_parameters, 0, "I1_soma", client)
        I2_neurons = rm.Neurons(rm.Circuit_Equation_Integrate_and_Fire_Soma, I_soma_parameters, 1, "I2_soma", client)
        I3_neurons = rm.Neurons(rm.Circuit_Equation_Integrate_and_Fire_Soma, I_soma_parameters, 2, "I3_soma", client)

        # The interface futures funciton is used to connect two neurons and build all the
        # components the connection consists of based on the paramter dict which contains
        # the parameters for each of the components in the connection.
        # THe function only produces futures objects though, which are not themselves connected yet
        #E1_neurons.interface_futures(self_E_1_parameter_dict, E1_neurons)
        #E1_neurons.interface_futures(EE_1_2_parameter_dict, E2_neurons)
        E1_neurons.interface_futures(EE_1_3_parameter_dict, E3_neurons)
        E1_neurons.interface_futures(EI_1_1_parameter_dict, I1_neurons)
        E1_neurons.interface_futures(EI_1_2_parameter_dict, I2_neurons)
        E1_neurons.interface_futures(EI_1_3_parameter_dict, I3_neurons)

        E2_neurons.interface_futures(self_E_2_parameter_dict, E2_neurons)
        E2_neurons.interface_futures(EE_2_1_parameter_dict, E1_neurons)
        E2_neurons.interface_futures(EE_2_3_parameter_dict, E3_neurons)
        E2_neurons.interface_futures(EI_2_1_parameter_dict, I1_neurons)
        E2_neurons.interface_futures(EI_2_2_parameter_dict, I2_neurons)
        E2_neurons.interface_futures(EI_2_3_parameter_dict, I3_neurons)

        E3_neurons.interface_futures(self_E_3_parameter_dict, E3_neurons)
        E3_neurons.interface_futures(EE_3_1_parameter_dict, E1_neurons)
        E3_neurons.interface_futures(EE_3_2_parameter_dict, E2_neurons)
        E3_neurons.interface_futures(EI_3_1_parameter_dict, I1_neurons)
        E3_neurons.interface_futures(EI_3_2_parameter_dict, I2_neurons)
        E3_neurons.interface_futures(EI_3_3_parameter_dict, I3_neurons)

        I1_neurons.interface_futures(self_I_1_parameter_dict, I1_neurons)
        I1_neurons.interface_futures(II_1_2_parameter_dict, I2_neurons)
        I1_neurons.interface_futures(II_1_3_parameter_dict, I3_neurons)
        I1_neurons.interface_futures(IE_1_1_parameter_dict, E1_neurons)
        I1_neurons.interface_futures(IE_1_2_parameter_dict, E2_neurons)
        I1_neurons.interface_futures(IE_1_3_parameter_dict, E3_neurons)

        I2_neurons.interface_futures(self_I_2_parameter_dict, I2_neurons)
        I2_neurons.interface_futures(II_2_1_parameter_dict, I1_neurons)
        I2_neurons.interface_futures(II_2_3_parameter_dict, I3_neurons)
        I2_neurons.interface_futures(IE_2_1_parameter_dict, E1_neurons)
        I2_neurons.interface_futures(IE_2_2_parameter_dict, E2_neurons)
        I2_neurons.interface_futures(IE_2_3_parameter_dict, E3_neurons)

        I3_neurons.interface_futures(self_I_3_parameter_dict, I3_neurons)
        I3_neurons.interface_futures(II_3_1_parameter_dict, I1_neurons)
        I3_neurons.interface_futures(II_3_2_parameter_dict, I2_neurons)
        I3_neurons.interface_futures(IE_3_1_parameter_dict, E1_neurons)
        I3_neurons.interface_futures(IE_3_2_parameter_dict, E2_neurons)
        I3_neurons.interface_futures(IE_3_3_parameter_dict, E3_neurons)

        # Since all components are currently future object we need to call the
        # get component results for the neurons to get the object proxies to be
        # able to call functions on them
        E1_neurons.get_component_results()
        E2_neurons.get_component_results()
        E3_neurons.get_component_results()
        I1_neurons.get_component_results()
        I2_neurons.get_component_results()
        I3_neurons.get_component_results()

        # the connect components function does what it says and connects the components
        # Internally this is done component by component because the internals of each component
        # is built when they are connected so the results must be gotten for each funciton call
        # before the next can be connected to it. Because of this we don't need to get the results for this call
        E1_neurons.connect_components()
        E2_neurons.connect_components()
        E3_neurons.connect_components()
        I1_neurons.connect_components()
        I2_neurons.connect_components()
        I3_neurons.connect_components()

        # Set the number of each cell type in each layer. Here we are setting the number of
        # inhibitory neurons to be 20% by killing 20% of the excitatory neurons in a layer
        # and setting the inhibitory neurons in the same layer to be living at the position
        # of the dead neurons in the excitatory layer.
        percentage_ratio = 0.2
        E1_dead_cells = ncp.random.uniform(0,1,population_size) <percentage_ratio
        I1_dead_cells = E1_dead_cells == 0
        E2_dead_cells = ncp.random.uniform(0,1,population_size) <percentage_ratio
        I2_dead_cells = E2_dead_cells == 0
        E3_dead_cells = ncp.random.uniform(0,1,population_size) <percentage_ratio
        I3_dead_cells = E3_dead_cells == 0

        future_E1 = E1_neurons.components[E1_neurons.name].set_dead_cells(E1_dead_cells)
        future_E2 = E2_neurons.components[E2_neurons.name].set_dead_cells(E2_dead_cells)
        future_E3 = E3_neurons.components[E3_neurons.name].set_dead_cells(E3_dead_cells)
        future_I1 = I1_neurons.components[I1_neurons.name].set_dead_cells(I1_dead_cells)
        future_I2 = I2_neurons.components[I2_neurons.name].set_dead_cells(I3_dead_cells)
        future_I3 = I3_neurons.components[I3_neurons.name].set_dead_cells(I3_dead_cells)

        future_E1.result()
        future_E2.result()
        future_E3.result()
        future_I1.result()
        future_I2.result()
        future_I3.result()

        # Create a list of the neuron populatiosn so that we can iterate over them when running the simulation
        neuron_populations = [E1_neurons, E2_neurons, E3_neurons, I1_neurons, I2_neurons, I3_neurons]
        '''
        Readout mechanisms
        '''

        P_delta_readout_parameters = {}
        P_delta_readout_parameters["nr_of_readout_neurons"] = 51
        P_delta_readout_parameters["error_tolerance"] = 0.05
        P_delta_readout_parameters["rho"] = 1 # squashing function boundries
        P_delta_readout_parameters["margin"] = 0.02
        P_delta_readout_parameters["clear_margins_importance"] = 1
        P_delta_readout_parameters["learning_rate"] = 0.0025
        P_delta_readout_parameters["temporal_upper_limit"] = upper_limit

        simulation_length = 200

        input_mask = ncp.random.uniform(0,1,population_size)
        input_mask = input_mask < 0.3
        input_mask = ncp.zeros(population_size)
        input_mask[:,0] = 1
        poisson_generator = rm.Poisson_Spike_Generator(1,1, 30)
        #new_input = ncp.zeros(population_size)
        #new_input[:,3] = 70
        #inputs.update_input(input_mask*70)
        #future.result()
        futures = []
        for nr, population in enumerate(neuron_populations):
            #if nr < 1:
            future = population.components[population.name].interface(inputs)
        for future in futures:
            future.result()

        '''
        Start simulation
        ######################################################################################
        '''
        v = ncp.zeros(simulation_length)
        spikes_plot = ncp.zeros(simulation_length)
        timeline = ncp.zeros(simulation_length)
        for t in range(simulation_length):
            # input spikes are generated
            spikes = poisson_generator.homogenous_poisson_spike(t*time_step)
            new_inputs = input_mask *20
            future = inputs.update_input(new_inputs)
            future.result()
            print(t*time_step, ncp.amax(inputs.interfacable))
            timeline[t] = (t+1 )*time_step

            for population in neuron_populations:
                population.update_current_values()

            for population in neuron_populations:
                population.get_results()


            for population in neuron_populations:
                population.compute_new_values()

            for population in neuron_populations:
                population.get_results()



            image_E1 = E1_neurons.components["E1_soma"].interfacable*255
            image_E2 = E2_neurons.components["E2_soma"].interfacable*255
            image_E3 = E3_neurons.components["E3_soma"].interfacable*255
            image_I1 = I1_neurons.components["I1_soma"].interfacable*255
            image_I2 = I2_neurons.components["I2_soma"].interfacable*255
            image_I3 = I3_neurons.components["I3_soma"].interfacable*255

            image = ncp.concatenate((image_E1, image_E2, image_E3, image_I1, image_I2, image_I3), axis = 0)
            #image = inputs.interfacable
            print(ncp.amax(image))
            image_shape = ncp.array(image.shape)
            image_shape *= 10
            image_shape = image_shape[::-1]
            image_shape = tuple(image_shape)

            #image = ncp.asnumpy(image)
            image = cv2.resize(image,image_shape)
            voltages = E1_neurons.components["E1_soma"].current_somatic_voltages
            v[t] = voltages[0,0]
            #u[t] = E1_neurons.components["E1_soma"].current_u[0,0]
            spikes_plot[t] = image_E1[0,0]

            #print(ncp.amax(image))
            image = image.astype(ncp.uint8)
            cv2.imshow('frame', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_simulation = True
                break
        for population in neuron_populations:
            population.get_results()
        cv2.destroyAllWindows()
        plt.plot(timeline, v, label = "somatic v")
        #plt.plot(timeline, u, label = "recovery u")
        plt.plot(timeline, spikes_plot, label = "spikes")
        plt.legend()
        plt.show()
