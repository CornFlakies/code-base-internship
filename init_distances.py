import argparse
import get_traj
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pair_disp_v2 import ParticleDispersion
from helper_functions import HelperFunctions as hp

# Load in image paths
argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('input_file', type=str)
args = argparser.parse_args()

df = pd.read_pickle(args.input_file) 
N = df.shape[1] // 2

pos = np.zeros((N, 2))
for n in range(N):     
    x = df.iloc[:, 2*n  ].values.astype(float)
    y = df.iloc[:, 2*n+1].values.astype(float)
    pos[n, 0] = x[0]
    pos[n, 1] = y[0]

# Define particle indices
xidx, yidx = np.meshgrid(np.arange(0, N), np.arange(0, N))

# Compute interparticle distances
diff_pos = pos[yidx] - pos[xidx]

# Get the interparticle distances, find the linked particles
dij = np.linalg.norm(diff_pos, axis=-1)       
dists = np.ravel(dij)
dists = dists[~np.isnan(dists)]

print(np.mean(dists) * 0.08)
