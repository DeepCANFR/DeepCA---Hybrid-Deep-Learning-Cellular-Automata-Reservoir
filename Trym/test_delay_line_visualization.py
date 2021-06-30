import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

spike_output = np.load("/home/trymlind/DeepCA---Hybrid-Deep-Learning-Cellular-Automata-Reservoir/Trym/DelayLine_test_1/test_DelayLine_test/delay_line/spike_output.npy")
spike_source = np.load("/home/trymlind/DeepCA---Hybrid-Deep-Learning-Cellular-Automata-Reservoir/Trym/DelayLine_test_1/test_DelayLine_test/delay_line/spike_source.npy")
print(spike_source.shape)
for i in range(spike_source.shape[0]):
    print()
    print("timestep: ",i)
    print(spike_source[i,:,:])
    print()
    print(spike_output[i,:,:])

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(spike_source[:,0,0])
ax2 = fig.add_subplot(212)
ax2.plot(spike_output[:,0,0])
fig.savefig('test.png')