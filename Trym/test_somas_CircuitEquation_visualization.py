import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

e_spike_data = np.load("/home/trymlind/DeepCA---Hybrid-Deep-Learning-Cellular-Automata-Reservoir/Trym/test_CircuitEquation_soma_test_1/E_I_CircuitEquation_network/I_CircuitEquation_somas_0/spiked_neurons.npy")
i_spike_data = np.load("/home/trymlind/DeepCA---Hybrid-Deep-Learning-Cellular-Automata-Reservoir/Trym/test_CircuitEquation_soma_test_1/E_I_CircuitEquation_network/I_CircuitEquation_somas_0/v.npy")
print(e_spike_data.shape)
for i in range(e_spike_data.shape[0]):
    print()
    print("timestep: ",i)
    print(e_spike_data[i,:,:])
    print()
    print(i_spike_data[i,:,:])

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(e_spike_data[:,0,0])
ax2 = fig.add_subplot(212)
ax2.plot(i_spike_data[:,0,0])
fig.savefig('test.png')