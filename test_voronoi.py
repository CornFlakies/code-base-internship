import numpy as np
from tqdm import tqdm
from scipy.spatial import Voronoi, ConvexHull
import matplotlib.pyplot as plt
from scipy.special import gamma
import argparse
import pandas as pd
from helper_functions import HelperFunctions as hp
from tqdm import tqdm

plt.close('all')


# SIMULATION OF M IMAGES OF N PARTICLES
M = 3072
N = 10
obs_vol = 1
V = []

# Load in image paths
argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('input_folder', type=str)
args = argparser.parse_args()

file_paths, _ = hp.load_images(args.input_folder, header='out')
    
# Loop over all the files
for file in file_paths:
    df = pd.read_pickle(file)
    nparticles = df.shape[1] // 2
    frames = df.shape[0]
    particle_list = np.zeros((frames, nparticles, 2))
    
    # Get particle x and y position
    for n in range(nparticles):     
        x = df.iloc[:, 2*n  ].values.astype(float)
        y = df.iloc[:, 2*n+1].values.astype(float)  
        particle_list[:, n, 0] = x
        particle_list[:, n, 1] = y 

for i in tqdm(range(M)):
    # Scrub nans from the dataset
    curr_frame = particle_list[i]
    idx = np.argwhere(np.isnan(curr_frame[:, 0]))
    curr_frame = np.delete(curr_frame, idx, axis=0)
    idx = np.argwhere(np.isnan(curr_frame[:, 1]))
    curr_frame = np.delete(curr_frame, idx, axis=0)
    
    vor = Voronoi(curr_frame)
        
    #plt.figure()
    #plt.plot(curr_frame[:, 0], curr_frame[:, 1], '.')
    #plt.show()

    regions_not = []
    empty_regions = []
    points_not = []
    regions_yes = []
    points_yes = []

    maxDF = 1023
    minDF = 0
    
    # VORONOI VOLUMES 
    for idx_input_point, reg_num in enumerate(vor.point_region): 
        indices = vor.regions[reg_num]
        vertices = vor.vertices
        cond = (-1 not in indices) and (all((vertices[indices]<maxDF).ravel())) and (all((vertices[indices]>minDF).ravel()))
        # there may be an empty region, bounded entirely by points at inifinity
        if not indices:
                regions_not.append(reg_num)
                empty_regions.append(reg_num)
                points_not.append(idx_input_point)

        elif cond:
                regions_yes.append(reg_num)
                points_yes.append(idx_input_point)

        # if -1 in indices then region is closed at infinity
        elif not cond:
                regions_not.append(reg_num)
                points_not.append(idx_input_point)
            
    N_ry = len(regions_yes)
    if N_ry > 0:
        Vols = np.zeros(N_ry)
        for nn, reg in enumerate(tqdm(regions_yes, desc='Computing volumes', disable=True, leave=False)):
                idx_vert_reg = vor.regions[reg]
                Vols[nn] = ConvexHull(vor.vertices[idx_vert_reg]).volume
        N_vols  = len(Vols)
        V.extend(Vols)

plt.figure()

Bins = np.geomspace(1e-2, 1e1, 35+1)
Bins_c = (Bins[:-1] + Bins[1:])/2

a, b, c = 4.806, 4.045, 1.168
x   = np.logspace(np.log10(Bins_c[0]), np.log10(Bins_c[-1]), 1000)
RPP = c*(b**(a/c))/gamma(a/c)*(x**(a-1))*np.exp(-b*(x**c))

# ------------------------------

Bins_c = Bins_c

xmin, xmax = 1e-3, 1e1
ymin, ymax = 1e-2, 2e0

# JFM limits
#xmin, xmax = 1e-4, 5e1
#ymin, ymax = 1e-3, 1e1


# ------------------------------
plt.loglog(x, RPP, '--', c='k', lw=2, label='RPP')

PDF, _ = np.histogram(V/np.mean(V), bins=Bins, density=True)

plt.plot(Bins_c, PDF, 'o')
plt.show()
