# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:03:01 2020

@author: trymlind
"""


import framework_module as fm
import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
Initiate model here
############################################################################################################
'''
# ex: neuron_poplulation_1 = fm.AS_Soma

'''
############################################################################################################
'''


cap = cv2.VideoCapture(0)
video = np.zeros((2,2))
while True:
    
    '''
    Update model here
    ########################################################################################################
    '''
    # If using video input, uncomment below code. 
    # frame is the video input from the camera
    # ret, frame = cap.read()
    
    
    
    # initialize video as the part you wish to visualize, ex neuron_population.spiked_neurons
    # video = output for visualiztion
    '''
    ########################################################################################################
    '''
    
    cv2.imshow('frame', video)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()