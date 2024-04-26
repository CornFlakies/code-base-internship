from helper_functions import HelperFunctions as hp
from scipy.spatial import Voronoi, ConvexHull, Delaunay
from scipy.special import gamma
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse


# Load in image paths
argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('input_folder', type=str)
args = argparser.parse_args()

file_paths, _ = hp.load_images(args.input_folder, header='out')
print(file_paths)

for file in file_paths:
    df = pd.read_pickle(file)
    Ntray = int(df.shape[1]//2)
    
    max = 0
    for n in range(Ntray):
        x = df.iloc[:, 2*n  ].values.astype(float)
        y = df.iloc[:, 2*n+1].values.astype(float)
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        
        if (len(x) > max):
            max = len(x)
    
    # SIMULATION OF M IMAGES OF N PARTICLES
    M = max
    N = Ntray
    obs_vol = 1
    V = []
    
    for i in tqdm(range(M)):
        points = []
    
        for n in range(Ntray):
            x = df.iloc[:, 2*n  ].values.astype(float)
            y = df.iloc[:, 2*n+1].values.astype(float)
            x = x[~np.isnan(x)]
            y = y[~np.isnan(y)]
            
            if not (len(x) <= i):
                points.append([x[i], y[i]])
    
        vor = Voronoi(points)
    
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
    
    Vtot.extend(V)


plt.figure()

Bins = np.geomspace(1e-2, 1e1, 50)
Bins_c = (Bins[:-1] + Bins[1:])/2

a, b, c = 4.806, 4.045, 1.168
x   = np.logspace(np.log10(bins_c[0]), np.log10(bins_c[-1]), 1000)
rpp = c*(b**(a/c))/gamma(a/c)*(x**(a-1))*np.exp(-b*(x**c))

a, b = 1.33, 1.41
x   = np.logspace(np.log10(bins_c[0]), np.log10(bins_c[-1]), 1000)
rpp = (b**a) / gamma(a) * (x**(a-1)) * np.exp(-b * x)
# ------------------------------

Bins_c = Bins_c

xmin, xmax = 1e-3, 1e1
ymin, ymax = 1e-2, 2e0

# JFM limits
#xmin, xmax = 1e-4, 5e1
#ymin, ymax = 1e-3, 1e1


# ------------------------------
#plt.plot(x, RPP, '--', c='k', lw=2, label='RPP')

PDF, _ = np.histogram(Vtot/np.mean(Vtot), bins=Bins, density=True)

plt.loglog(Bins_c, PDF, '.-')
plt.show()

