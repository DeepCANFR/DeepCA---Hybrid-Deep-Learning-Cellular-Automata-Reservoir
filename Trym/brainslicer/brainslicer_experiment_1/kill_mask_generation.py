import numpy as np

nr_of_experiments = 1
population_size = [10,100]
nr_of_layers = 3
percentage_ratio = 0.2


for experiment_nr in range(nr_of_experiments):
    for layer_nr in range(nr_of_layers):
        filename_E = "experiment_" + str(experiment_nr) + "_layer_" + str(layer_nr) + "_Excitatory_kill_mask"
        filename_I = "experiment_" + str(experiment_nr) +  "_layer_" + str(layer_nr) + "_Inhibitory_kill_mask"

        E_kill_mask = np.random.uniform(0,1,population_size) < percentage_ratio
        I_kill_mask = E_kill_mask == 0

        np.save(filename_E, E_kill_mask)
        np.save(filename_I, I_kill_mask)
