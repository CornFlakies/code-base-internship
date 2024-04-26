from helper_functions import HelperFunctions as hp
from scipy.spatial import ConvexHull, Delaunay
from scipy.special import gamma
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import os

# Load in image paths
argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('input_folder', type=str)
args = argparser.parse_args()
save = False

# Get all the stuff in the supplied directory, and define the files that needed to be checked
directory = os.listdir(args.input_folder)
files_to_check = ('del_all_areas.npy', 'del_area_count.npy', 'del_particles_per_frame.npy')

# Define the relevant arrays
all_areas = []
area_count = []
particle_count = []

# Check if directory contains precalculated npy files, we can skip the computations then
if all(file in "".join(directory) for file in files_to_check):
    save = False
    campaign_folders = []
    for file_indir in directory:
        print(file_indir)
        if (files_to_check[0] in file_indir):
            all_areas.append(np.load(os.path.join(args.input_folder, file_indir)))
        elif (files_to_check[1] in file_indir):
            area_count.append(np.load(os.path.join(args.input_folder, file_indir)))
        elif (files_to_check[2] in file_indir):
            particle_count.append(np.load(os.path.join(args.input_folder, file_indir)))

# If multiple campaigns are supplied, loop over all those directories
elif (args.input_folder.split(os.sep)[-2] == 'all_campaign_dfs'):
    campaign_folders = hp.load_folders(args.input_folder)
    save = True
# If only one df is supplied, just compute that one
else:
    campaign_folders = [args.input_folder]
    save = True

for folder in campaign_folders:
    # Load the df
    file_paths, _ = hp.load_images(folder, header='out')
    
    # Define lists to store the relevant data
    area_per_campaign = []
    particles_per_campaign = []
    area_count_per_campaign = []

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
    
        # Loop over all frames in the dataset
        for f in tqdm(range(frames)): 
            # Scrub nans from the dataset
            curr_frame = particle_list[f]
            idx = np.argwhere(np.isnan(curr_frame[:, 0]))
            curr_frame = np.delete(curr_frame, idx, axis=0)
            idx = np.argwhere(np.isnan(curr_frame[:, 1]))
            curr_frame = np.delete(curr_frame, idx, axis=0)
            
            # Create Delaunay object
            dela = Delaunay(curr_frame)
            vertices = curr_frame[dela.simplices]
    
            # Get triangle areas
            areas_per_frame = []
            min_vertices = []
            for i, vertex in enumerate(vertices):
                to_det = np.concatenate((vertex, np.ones((1, np.size(vertex, axis=0))).T), axis=1)
                area = 0.5 * np.linalg.det(to_det) 
                area_per_campaign.append(area)                

            # Find the volumes of the remaining volumes if they exist and save the total amount of particles in the system
            closed_regions = len(vertices)
            area_count_per_campaign.append(closed_regions)
            particles_per_campaign.append(np.size(curr_frame, axis=0))

    area_per_campaign = np.array(area_per_campaign)
    all_areas.append(area_per_campaign)
    area_count.append(area_count_per_campaign)
    particle_count.append(particles_per_campaign)

    # Save created files if they do not exist yet
    if save:
        filename = files_to_check[0]
        file_destination = os.path.join(args.input_folder, folder.split(os.sep)[-1] + '_' + filename)
        np.save(file_destination, area_per_campaign)
        filename = files_to_check[1] 
        file_destination = os.path.join(args.input_folder, folder.split(os.sep)[-1] + '_' + filename)
        np.save(file_destination, area_count_per_campaign)
        filename = files_to_check[2] 
        file_destination = os.path.join(args.input_folder, folder.split(os.sep)[-1] + '_' + filename)
        np.save(file_destination, particles_per_campaign)         

# Define total area of the box
Lx = 700
Ly = 800
A_tot = Lx * Ly

# Plot that boi
bins = np.geomspace(1e-2, 4, 50)
bins_c = (bins[:-1] + bins[1:]) / 2

plt.figure()
for area, particles in zip(all_areas, particle_count):  
    # Get the normalized areas 
    all_normalized_areas = []
    indices = np.concatenate((np.array([0]), np.cumsum(particles)))
    for i in range(len(indices) - 1):
        A_avg = A_tot / particles[i] 
        all_normalized_areas.append(area[indices[i]:indices[i + 1]] / A_avg)
    
    all_normalized_areas = np.concatenate(all_normalized_areas).ravel()
    PDF, bins = np.histogram(all_normalized_areas, bins=bins, density=True)    
    plt.loglog(bins_c, PDF, 'o')

a, b = 1.34, 1.42
x   = np.logspace(np.log10(1e-2), np.log10(4), 1000)
rpp = (b**a) / gamma(a) * (x**(a-1)) * np.exp(-b * x)

plt.loglog(x, rpp, '--')
plt.legend()
plt.grid()

bins   = np.geomspace(1e-2, 10, 50)
bins   = np.linspace(1e-2, 10, 100)
bins_c = (bins[:-1] + bins[1:]) / 2
x      = np.logspace(np.log10(bins_c[0]), np.log10(bins_c[-1]), 100)

fig, ax = plt.subfigures(nrows=2, ncols=2)
i = 0
for area, particles in zip(all_areas, particle_count):  
    idx = np.unravel_index(i, (2, 2))

    # Get the normalized areas 
    all_normalized_areas = []
    indices = np.concatenate((np.array([0]), np.cumsum(particles)))
    for i in range(len(indices) - 1):
        A_avg = A_tot / particles[i] 
        all_normalized_areas.append(area[indices[i]:indices[i + 1]] / A_avg)
    
    all_normalized_areas = np.concatenate(all_normalized_areas).ravel()
    PDF, _ = np.histogram(all_normalized_areas, bins=bins, density=True)    

    mean = np.mean(all_normalized_areas)
    std  = np.std(all_normalized_areas)
    a = mean**2 / std**2 
    b = mean / std**2
    rpp = (b**a) / gamma(a) * (x**(a - 1)) * np.exp(-b * x)
    
    ax[i].semilogy(bins[:-1], PDF, 'x')
    ax[i].semilogy(x, rpp, '--')
    ax[i].grid()

plt.show()
