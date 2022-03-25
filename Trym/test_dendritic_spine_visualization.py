import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

synaptic_input = np.load("/home/trymlind/DeepCA---Hybrid-Deep-Learning-Cellular-Automata-Reservoir/Trym/test_DendriticSpines_1/Arborizer_test/dendritic_spine/synaptic_input.npy")
synaptic_output = np.load("/home/trymlind/DeepCA---Hybrid-Deep-Learning-Cellular-Automata-Reservoir/Trym/test_DendriticSpines_1/Arborizer_test/dendritic_spine/synaptic_output.npy")
print(synaptic_input.shape)
for i in range(synaptic_input.shape[0]):
    print()
    print("timestep: ",i)
    print(synaptic_input[i,:,:,:])
    print()
    print(synaptic_output[i,:,:,:])

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(synaptic_input[:,0,0,0])
ax2 = fig.add_subplot(212)
ax2.plot(synaptic_output[:,0,0,0])
fig.savefig('test.png')