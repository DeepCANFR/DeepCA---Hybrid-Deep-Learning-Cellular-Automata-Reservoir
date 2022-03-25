import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

connection_array = np.load("/home/trymlind/DeepCA---Hybrid-Deep-Learning-Cellular-Automata-Reservoir/Trym/test_Arborizer_1/Arborizer_test/arborizer/connection_array.npy")
spike_source = np.load("/home/trymlind/DeepCA---Hybrid-Deep-Learning-Cellular-Automata-Reservoir/Trym/test_Arborizer_1/Arborizer_test/arborizer/spike_source.npy")
print(spike_source.shape)
for i in range(spike_source.shape[0]):
    print()
    print("timestep: ",i)
    print(spike_source[i,:,:])
    print()
    print(connection_array[i,:,:,:])

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(connection_array[:,0,0])
ax2 = fig.add_subplot(212)
ax2.plot(connection_array[:,0,1,0])
fig.savefig('test.png')