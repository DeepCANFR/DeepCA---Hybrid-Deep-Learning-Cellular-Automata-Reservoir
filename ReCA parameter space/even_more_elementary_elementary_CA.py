# Elementary CA includes the CA that exclude computation from one or more of the cells.
# this is a short program to find the 3 sets that do this for left central and right nodes.
# this reduces the ECA to even smaller computational universes

import numpy as np
import helpers.helper as helper

ignores_left = []
ignores_central = []
ignores_right = []

for i in range(0, 256):
    binary = helper.int_to_binary_string(i, 8)
    print(i, binary)
    if binary[0] == binary[4] and binary[1] == binary[5] and binary[2] == binary[6] and binary[3] == binary[7]:
        ignores_left.append(i)
    if binary[0] == binary[2] and binary[1] == binary[3] and binary[4] == binary[6] and binary[5] == binary[7]:
        ignores_central.append(i)
    if binary[0] == binary[1] and binary[2] == binary[3] and binary[4] == binary[5] and binary[6] == binary[7]:
        ignores_right.append(i)

print("left set:", ignores_left)
print("central set:", ignores_central)
print("right set:", ignores_right)