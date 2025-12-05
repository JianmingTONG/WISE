#!/home/jyh/project/winograd/winograd-venv/bin/python3
# run  := python3 plot.py
# dir  := .
# kid  :=

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

coord = np.load("./my.npy")
# coord = np.load("./data.npy")
coord = np.array(coord, dtype=np.int32)
rows = coord[:, 0].astype(np.int32)
cols = coord[:, 1].astype(np.int32)

n_slots = 1 << 9
T_in = 32
T_out = 8
H = T_out * n_slots
W = T_in * n_slots

fig, ax = plt.subplots()

h_lines = [([0, W], [i*n_slots, i*n_slots]) for i in range(T_out+1)]
v_lines = [([i*n_slots, i*n_slots], [0, H]) for i in range(T_in+1)]
segs = [[(x1, y1), (x2, y2)] for (xs, ys) in h_lines+v_lines for (x1, x2), (y1, y2) in [(xs, ys)]]
ax.add_collection(LineCollection(segs, colors="black", linewidths=1, antialiased=False))

ax.scatter(cols, rows, s=10, c="red", marker='.', linewidths=0, rasterized=True)

ax.set_xlim(0, W)
ax.set_ylim(H, 0)

ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

