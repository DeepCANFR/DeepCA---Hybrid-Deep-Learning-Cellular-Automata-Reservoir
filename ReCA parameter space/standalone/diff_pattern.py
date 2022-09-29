import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

width = 50
height_fig = 25
fargs_list = [(a,) for a in [110]]

init = np.random.randint(2, size=(width, 1)).astype(np.float64)
init_m = np.array(init, copy=True)
if init_m[int(width/2),] == 1.0:
    init_m[int(width/2),] = 0.0
else:
    init_m[int(width/2),] = 1.0

exp = experiment.Experiment()
g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin', init=init)
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx)

exp.add_connection("g_ca_conn",
                   connection.WeightedConnection(g_ca_bin,g_ca_bin,
                                                 act.rule_binary_ca_1d_width3_func,
                                                 g_ca_bin_conn, fargs_list=fargs_list))

exp.initialize_cells()

im_ca_n = np.zeros((height_fig, width))
im_ca_n[0] = exp.get_group_cells_state("g_ca", "g_ca_bin")[:, 0]

for i in range(1, height_fig):
    exp.run_step()
    im_ca_n[i] = exp.get_group_cells_state("g_ca", "g_ca_bin")[:, 0]

exp.close()


exp = experiment.Experiment()
g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin', init=init_m)
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx)

exp.add_connection("g_ca_conn",
                   connection.WeightedConnection(g_ca_bin,g_ca_bin,
                                                 act.rule_binary_ca_1d_width3_func,
                                                 g_ca_bin_conn, fargs_list=fargs_list))

exp.initialize_cells()

im_ca_m = np.zeros((height_fig, width))
im_ca_m[0] = exp.get_group_cells_state("g_ca", "g_ca_bin")[:, 0]

for i in range(1, height_fig):
    exp.run_step()
    im_ca_m[i] = exp.get_group_cells_state("g_ca", "g_ca_bin")[:, 0]

exp.close()


im_ca = np.array(im_ca_n, copy=True)
for i, j in np.ndindex(im_ca.shape):
    if im_ca[i,j] != im_ca_m[i,j]:
        im_ca[i, j] = 2.0



cmap = colors.ListedColormap(["white", "lightgray", "darkred"])
fig, axes = plt.subplots(1, 3)

axes[0].imshow(im_ca_n, cmap="binary")
axes[0].axis('off')
axes[1].imshow(im_ca_m, cmap="binary")
axes[1].axis('off')
axes[2].imshow(im_ca, cmap=cmap)
axes[2].axis('off')



# plt.title(f'Rule {fargs_list[0][0]}, {height_fig} step')

plt.show()


# fig.savefig(f'diff.png', bbox_inches="tight", dpi=300)



