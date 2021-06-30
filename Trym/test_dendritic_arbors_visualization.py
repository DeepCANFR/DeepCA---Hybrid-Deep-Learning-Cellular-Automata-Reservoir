import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

synaptic_response = np.load("/home/trymlind/DeepCA---Hybrid-Deep-Learning-Cellular-Automata-Reservoir/Trym/test_DendriticArbors_1/Arborizer_test/axonal_spine/synaptic_response.npy")
spike_source = np.load("/home/trymlind/DeepCA---Hybrid-Deep-Learning-Cellular-Automata-Reservoir/Trym/test_DendriticArbors_1/Arborizer_test/delay_line/spike_output.npy")
print(spike_source.shape)
for i in range(spike_source.shape[0]):
    print()
    print("timestep: ",i)
    print(spike_source[i,:,:])
    print()
    print(synaptic_response[i,:,:,:])

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(spike_source[:,0,0])
ax2 = fig.add_subplot(212)
ax2.plot(synaptic_response[:,0,0,0])
fig.savefig('test.png')