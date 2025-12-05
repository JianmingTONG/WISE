#!/home/jyh/project/winograd/winograd-venv/bin/python3
# run  := python3 check.py
# dir  := .
# kid  :=

import pickle
import numpy as np

with open("./final.pkl", "rb") as f:
    obj = pickle.load(f)

scores = obj.flatten()[:1000]
idx = np.argsort(scores)[-20:][::-1]
print(idx)
