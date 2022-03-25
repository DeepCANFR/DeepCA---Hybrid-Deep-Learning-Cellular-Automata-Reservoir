import numpy as np
import analysis_module as am
import matplotlib.pyplot as plt
network_history_test_1 = []
network_history_test_2 = []
network_history_test_3 = []
for i in range(1):
    #filename = str(i)+"network_history_1_Izhikevich_injection_at_510.npy"
    filename = "train_nr_0_network_history_1_Izhikevich.npy"
    #filename = str(i)+"network_history_1_Izhikevich_injection_at_510.npy"
    base_train_network_history = np.load(filename)
    #network_history_test_1.append(np.load(filename))
    '''
    filename = str(i)+"network_history_test_2.npy"
    network_history_test_2.append(np.load(filename))
    filename = str(i)+"network_history_test_3.npy"
    network_history_test_3.append(np.load(filename))
    '''

for i in range(1):
    #filename = str(i)+"network_history_1_Izhikevich_injection_at_500.npy"
    #filename = str(i)+"network_history_1_Izhikevich_injection_at_510.npy"
    filename = "train_nr_1_network_history_1_Izhikevich.npy"
    jittered_train_network_history = np.load(filename)
    #network_history_test_1.append(np.load(filename))
    '''
    filename = str(i)+"network_history_jittered_2.npy"
    network_history_test_2.append(np.load(filename))
    filename = str(i)+"network_history_jittered_3.npy"
    network_history_test_3.append(np.load(filename))
    '''

convoled_jittered_train = np.zeros(jittered_train_network_history.shape)
convoled_base_train = np.zeros(base_train_network_history.shape)



for index_1 in range(base_train_network_history.shape[0]):
    for index_2 in range(base_train_network_history.shape[1]):
        for layer_nr in range(convoled_jittered_train.shape[2]):
            convoled_jittered_train[index_1, index_2, layer_nr, :] =  am.gaussian_convolution(jittered_train_network_history[index_1, index_2, layer_nr, :])
            convoled_base_train[index_1, index_2, layer_nr, :] = am.gaussian_convolution(base_train_network_history[index_1, index_2, layer_nr, :])

print(jittered_train_network_history.shape)
print(base_train_network_history.shape)
state_shape = np.product(base_train_network_history.shape[:-1])
time_shape = base_train_network_history.shape[-1]

base_train_network_history_flattened = np.zeros((state_shape,time_shape ))
jittered_train_network_history_flattened = np.zeros((state_shape,time_shape ))
for i in range(time_shape):
    base_train_network_history_flattened[:,i] = convoled_base_train[:,:,:,i].flatten()
    jittered_train_network_history_flattened[:,i] = convoled_jittered_train[:,:,:,i].flatten()

    #base_train_network_history_flattened[:,i] = base_train_network_history[:,:,:,i].flatten()
    #jittered_train_network_history_flattened[:,i] = jittered_train_network_history[:,:,:,i].flatten()

am.plot_distanceplane(base_train_network_history_flattened, jittered_train_network_history_flattened)

basic_euclidian_distance = np.linalg.norm(base_train_network_history_flattened-jittered_train_network_history_flattened, ord = 2, axis = 0)
basic_overlap = np.sum((base_train_network_history_flattened + jittered_train_network_history_flattened), axis = 0)

plt.figure()
plt.plot(basic_euclidian_distance)
plt.figure()
plt.plot(basic_overlap)
plt.figure()
excitatory_rate = base_train_network_history[:,:,0:3,:]
inhibitory_rate = base_train_network_history[:,:,3:6,:]
excitatory_rate_sum = np.sum(excitatory_rate,axis = (0,1,2))
inhibitory_rate_sum = np.sum(inhibitory_rate,axis = (0,1,2))
plt.plot(excitatory_rate_sum)
plt.plot(inhibitory_rate_sum)

excitatory_rate = jittered_train_network_history[:,:,0:3,:]
inhibitory_rate = jittered_train_network_history[:,:,3:6,:]
excitatory_rate_sum = np.sum(excitatory_rate,axis = (0,1,2))
inhibitory_rate_sum = np.sum(inhibitory_rate,axis = (0,1,2))
plt.plot(excitatory_rate_sum)
plt.plot(inhibitory_rate_sum)

plt.show()
