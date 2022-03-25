import cv2
import numpy as np
import time

for i in range(1):
    #filename = str(i)+"network_history_test_1.npy"
    filename = str(i)+"network_history_1_Izhikevich_injection_at_510.npy"
    base_train_network_history = np.load(filename)
    #network_history_test_1.append(np.load(filename))
    '''
    filename = str(i)+"network_history_test_2.npy"
    network_history_test_2.append(np.load(filename))
    filename = str(i)+"network_history_test_3.npy"
    network_history_test_3.append(np.load(filename))
    '''

for i in range(1):
    #filename = str(i)+"network_history_jittered_1.npy"
    filename = str(i)+"network_history_1_Izhikevich_injection_at_500.npy"
    jittered_train_network_history = np.load(filename)
    #network_history_test_1.append(np.load(filename))
    '''
    filename = str(i)+"network_history_jittered_2.npy"
    network_history_test_2.append(np.load(filename))
    filename = str(i)+"network_history_jittered_3.npy"
    network_history_test_3.append(np.load(filename))
    '''
population_y = base_train_network_history.shape[0]
population_x = base_train_network_history.shape[1]
image = np.zeros((int(population_y*3), int(population_x), 3))

for t in range(base_train_network_history.shape[-1]):
    for layer_nr in range(3):
        image_base_E = base_train_network_history[:,:,layer_nr,t]
        image_jittered_E = jittered_train_network_history[:,:,layer_nr, t]
        image_base_I = base_train_network_history[:,:,layer_nr+3,t]
        image_jittered_I = jittered_train_network_history[:,:,layer_nr+3,t]


        summed_E = (image_base_E + image_jittered_E)
        summed_I = image_base_I + image_jittered_I
        total_summed = summed_E + summed_I
        image_diff = total_summed == 1
        image_E_same = summed_E == 2
        image_I_same = summed_I == 2

        if layer_nr == 0:
            image_E_out = summed_E > 0
            image_I_out = summed_I > 0
            image_diff_out = image_diff
        else:
            image_E_out = np.concatenate((image_E_out, summed_E>0), axis = 0)
            image_I_out = np.concatenate((image_I_out, summed_I>0), axis =0)
            image_diff_out = np.concatenate((image_diff_out, image_diff), axis = 0)
    image[:,:,2] = image_E_out
    image[:,:,0] = image_I_out
    image[:,:,1] = image_diff_out

    image_shape = np.array(image.shape[0:2])
    image_shape *= 15
    image_shape = image_shape[::-1]
    image_shape = tuple(image_shape)

    #image = ncp.asnumpy(image)
    image_out = cv2.resize(image,image_shape)
    image_out *= 255
    image_out = image_out.astype(np.uint8)
    time.sleep(0.2)
    print(t)
    cv2.imshow('frame', image_out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_simulation = True
        break
