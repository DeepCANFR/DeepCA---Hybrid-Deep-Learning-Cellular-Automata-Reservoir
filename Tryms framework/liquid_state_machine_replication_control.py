# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:00:11 2021

@author: trymlind
"""


import realistic_module_test as rm
import matplotlib.pyplot as plt
import tensorflow as tf

try:
    import cupy as cp
    cupy = True
except:
    print("Failed to import cupy, attempting to import numpy instead")
    cupy = False
    import numpy as cp
    
import numpy as np
import cv2

'''
Load stimuli
#######################################################################################
'''
(image_train, label_train), (image_test, label_test) = tf.keras.datasets.mnist.load_data()

P_delta_readout_parameters = {}
P_delta_readout_parameters["nr_of_readout_neurons"] = 53
P_delta_readout_parameters["error_tolerance"] = 0.5
P_delta_readout_parameters["rho"] = 1 # squashing function boundries
P_delta_readout_parameters["margin"] = 0.5
P_delta_readout_parameters["clear_margins_importance"] = 0.1
P_delta_readout_parameters["learning_rate"] = 0.00025

readout_list = []
for class_nr in range(10):
    readout_list.append(rm.Readout_P_Delta(P_delta_readout_parameters))
    
    #readout_list.append(rm.Readout_P_Delta_prototype_learner(P_delta_readout_parameters))
    readout_list[class_nr].interface_read_variable(image_train[0,:,:])

image = image_train[0,:,:]
image_shape = np.array(image.shape)
image_shape *= 5
#image_shape = image_shape[::-1]
image_shape = tuple(image_shape)
image = cv2.resize(image,image_shape)

total_image = np.zeros((image.shape[0]*10, image.shape[1]*4))
nr_of_epochs = 4000

for epoch in range(nr_of_epochs):
    print("Epoch nr: ",epoch)
    for image_nr in range(len(image_train)):
        for class_nr, readout in enumerate(readout_list):
            desired_output = label_train[image_nr] == class_nr#label_train[image_nr] == class_nr
            #print("class nr: ", class_nr, "label: ", label_train[image_nr], "desired output: ", desired_output)
            if desired_output == 0:
                desired_output = -1
            #print(desired_output)
            readout.inputs = cp.array(image_train[image_nr,:,:])/255
            readout.update_weights(desired_output)
            #print("current_class: ", label_train[image_nr], "readout ",class_nr, "output: ", readout.current_population_output)
            
            
            if cupy:
                #image= cp.asnumpy(excitatory_somas.current_spiked_neurons*255)
                image_pos_weights = readout.weights > 0
                image_pos_weights = readout.weights * image_pos_weights
                image_pos_weights = (cp.asnumpy(cp.sum(image_pos_weights, axis = 2)))
                pos_max = cp.amax(image_pos_weights)
                if pos_max > 0:
                    image_pos_weights /= pos_max
                
                image_neg_weights = readout.weights < 0
                image_neg_weights = readout.weights * image_neg_weights*-1
                image_neg_weights = (cp.asnumpy(cp.sum(image_neg_weights, axis = 2)))
                neg_max = cp.amax(image_neg_weights)
                if neg_max > 0:
                    image_neg_weights /= neg_max
                
                
                image = np.concatenate((image_pos_weights, image_neg_weights,(cp.asnumpy(cp.sum(readout.weights, axis = 2))), image_train[image_nr]), axis = 1)
            else:
                image= cp.sum(readout.weights, axis = 2)
            
            
            image_shape = np.array(image.shape)
            image_shape *= 5
            image_shape = image_shape[::-1]
            image_shape = tuple(image_shape)
            image = cv2.resize(image,image_shape)
             
            total_image[ image.shape[0]*class_nr: image.shape[0]*class_nr + image.shape[0],:] = image
          
        
            if class_nr == 9:
                cv2.imshow('frame', total_image)
            #print(readout.weights)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    classifications = np.zeros(10)
    correct_classifcations = 0
    
    
    for test_image_nr in range(len(image_test)):
        for index, readout in enumerate(readout_list):
            classifications[index] = readout.classify(cp.array(image_test[test_image_nr])/255)
        correct_classifcations += classifications.argmax() == label_test[test_image_nr]
    print("Accuracy; ",correct_classifcations/(len(image_test )))

cv2.destroyAllWindows()

