import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


def getCA(rule, width, hight, init):
    fargs_list = [(a,) for a in [rule]]
    exp = experiment.Experiment()
    g_ca = exp.add_group_cells(name="g_ca", amount=width)
    neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
    g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin', init=init)
    g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn', width, \
                                               neighbors=neighbors, \
                                               center_idx=center_idx)

    exp.add_connection("g_ca_conn",
                       connection.WeightedConnection(g_ca_bin, g_ca_bin,
                                                     act.rule_binary_ca_1d_width3_func,
                                                     g_ca_bin_conn, fargs_list=fargs_list))

    exp.initialize_cells()

    im_ca = np.zeros((hight, width))
    im_ca[0] = exp.get_group_cells_state("g_ca", "g_ca_bin")[:, 0]

    for i in range(1, hight):
        exp.run_step()
        im_ca[i] = exp.get_group_cells_state("g_ca", "g_ca_bin")[:, 0]

    exp.close()
    return im_ca


width = 200
height = 120
single = np.zeros((width, 1)).astype(np.float64)
single[0][0] = 1.0

ca_1 = getCA(90, width, height, single)
ca_2 = getCA(90, width, height, "random")
singlePattern = ca_1[-1]
randomInit = ca_2[0]
pattern = np.zeros(width).astype(np.float64)
for cellIndex in range(0, width):
    shape = np.roll(singlePattern, cellIndex)
    if randomInit[cellIndex] == 1.0:
        pattern = pattern + shape
print("matching:")
print((ca_2[-1] == (pattern % 2)).all())


ca_3 = [randomInit, pattern % 2]

fig, axes = plt.subplots(1, 3)
plt.subplots_adjust(wspace=0.1, hspace=0.2)
csfont = {'fontname': 'calibri'}

axes[0].imshow(ca_1, cmap="binary")
axes[0].axis('off')
axes[0].set_title("CA central", fontdict=csfont)
axes[1].imshow(ca_2, cmap="binary")
axes[1].axis('off')
axes[1].set_title("CA random", fontdict=csfont)
axes[2].imshow(ca_3, cmap="binary")
axes[2].axis('off')
axes[2].set_title("CA predicted", fontdict=csfont)

plt.show()

# fig.savefig(f'reducible.png', bbox_inches="tight", dpi=300)
