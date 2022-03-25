import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2

matplotlib.use('Agg')

e_spike_data = np.load("/home/trymlind/DeepCA---Hybrid-Deep-Learning-Cellular-Automata-Reservoir/Trym/test_2d_culture_3_neuron_types/E_I_network/excitatory_neuron/soma/spiked_neurons.npy")
i_spike_data = np.load("/home/trymlind/DeepCA---Hybrid-Deep-Learning-Cellular-Automata-Reservoir/Trym/test_2d_culture_3_neuron_types/E_I_network/inhibitory_neuron/soma/spiked_neurons.npy")
print(e_spike_data.shape)

fig = plt.figure()
ax1 = fig.add_subplot(411)
ax1.imshow(e_spike_data[100,:,:])
ax2 = fig.add_subplot(412)
ax2.imshow(e_spike_data[110,:,:])
ax3 = fig.add_subplot(421)
ax3.imshow(e_spike_data[510,:,:])
ax4 = fig.add_subplot(422)
ax4.imshow(e_spike_data[1010,:,:])
fig.savefig('test.png')


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("2d_culture_spikes_2_excitatory_reverse_I.avi",fourcc, 20.0, (500,500))
frame = np.zeros((3,100,100))
for i in range(e_spike_data.shape[0]):
    vis = e_spike_data[i,:,:].astype(np.uint8)*255
    vis2 = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    out.write(vis2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("2d_culture_spikes_2_inhibitory_reverse_I.avi",fourcc, 20.0, (500,500))
frame = np.zeros((3,100,100))
for i in range(i_spike_data.shape[0]):
    vis = i_spike_data[i,:,:].astype(np.uint8)*255
    vis2 = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    out.write(vis2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()