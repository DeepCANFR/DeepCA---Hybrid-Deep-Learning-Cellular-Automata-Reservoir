# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 00:51:51 2021

@author: trymlind
"""
correct_classifcations = 0

for test_image_nr in range(len(image_test)):
    classifications = cp.zeros(10)
    for index, readout in enumerate(readout_list):
        classifications[index] = readout.classify(cp.array(image_test[test_image_nr,:,:])/255)
        #readout.update_weights(cp.array(image_test[test_image_nr,:,:])/255)
        #classifications[index] = readout.current_population_output
        #print(classifications)
    correct_classifcations += classifications.argmax() == label_test[test_image_nr]
    #print(classifications.argmax(),label_test[test_image_nr])
    #print(classifications)
    print("Accuracy; ",correct_classifcations/(test_image_nr + 1))
