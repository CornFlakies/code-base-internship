from helper_functions import HelperFunctions as hp
from scipy.spatial import ConvexHull, Voronoi
from scipy.signal import savgol_filter 
from scipy.special import gamma
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import os
import re

# Gamma distribution with c parameter
def rpp_c(x, a, b, c):
    return c*(b**(a/c))/gamma(a/c)*(x**(a-1))*np.exp(-b*(x**c)) 

# Gamma distribution
def rpp(x, a, b):
    return (b**a) / gamma(a) * (x**(a - 1)) * np.exp(-b * x)

# Load in image paths
argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('input_folder', type=str)
args = argparser.parse_args()
save = False

# Get all the stuff in the supplied directory, and define the files that needed to be checked
directory = os.listdir(args.input_folder)
# Very involved sorting thingy, but it makes sure that the files are sorted based on forcing amplitude
directory = sorted(directory, key = lambda folder: re.search(r'\d+', folder).group().zfill(2))
files_to_check = ('vor_all_areas.npy', 'vor_area_count.npy', 'vor_particles_per_frame.npy')

# Define the relevant arrays
all_areas = []
area_count = []
particle_count = []

maxDF = 700
minDF = 200

# Check if directory contains precalculated npy files, we can skip the computations then
if all(file in "".join(directory) for file in files_to_check):
    save = False
    campaign_folders = []
    for file_indir in directory:
        if ((files_to_check[0] in file_indir) & (file_indir is not None)):
            print(file_indir)
            all_areas.append(np.load(os.path.join(args.input_folder, file_indir)))
        elif ((files_to_check[1] in file_indir) & (file_indir is not None)):
            area_count.append(np.load(os.path.join(args.input_folder, file_indir)))
        elif ((files_to_check[2] in file_indir) & (file_indir is not None)):
            particle_count.append(np.load(os.path.join(args.input_folder, file_indir)))

# If multiple campaigns are supplied, loop over all those directories
elif (args.input_folder.split(os.sep)[-2] == 'all_campaign_dfs'):
    campaign_folders = hp.load_folders(args.input_folder)
    save = True
# If only one df is supplied, just compute that one
else:
    campaign_folders = [args.input_folder]
    save = True

# Loop over all folders 
for folder in sorted(campaign_folders, key = lambda folder: re.search(r'\d+', folder).group().zfill(2)):
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
            
            # Create Voronoi object
            vor = Voronoi(curr_frame)
        
            # Defining lists used for removing outlying volumes
            regions_not = []
            empty_regions = []
            points_not = []
            regions_yes = []
            points_yes = []
                
            # Remove the outliers in the voronoi object 
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
                                
        
            # Find the volumes of the remaining volumes if they exist and save the total amount of particles in the system
            closed_regions = len(regions_yes)
            particles_per_campaign.append(np.size(curr_frame, axis=0))

            if closed_regions > 0:
                vols = np.zeros(closed_regions)
                for nn, reg in enumerate(tqdm(regions_yes, desc='Computing volumes', disable=True, leave=False)):
                        idx_vert_reg = vor.regions[reg]
                        vols[nn] = ConvexHull(vor.vertices[idx_vert_reg]).volume
                v_vols = len(vols)
                area_per_campaign.append(vols)
                area_count_per_campaign.append(v_vols)

    # Add all the voronoi data to lists storing all the campaigns
    area_per_campaign = np.concatenate(area_per_campaign).ravel()
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


# -----------------------------------------------------------------------------------------
# Plot the average pdfs of all voronoi diagrams for all forcing strenghts
# -----------------------------------------------------------------------------------------
# Define total area of the box
Lx = maxDF - minDF
Ly = Lx
A_tot = Lx * Ly
Alist = ['A5', 'A10', 'A15', 'A20']

# Plot that boi
bins = np.geomspace(1e-2, 4, 50)
bins_c = (bins[:-1] + bins[1:]) / 2

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
ii = 0
for area, particles in zip(all_areas, particle_count):  
    # Get the normalized areas 
    all_normalized_areas = []
    indices = np.concatenate((np.array([0]), np.cumsum(particles)))
    for i in range(len(indices) - 1):
        A_avg = A_tot / particles[i]
        all_normalized_areas.append(area[indices[i]:indices[i + 1]] / A_avg)
        
    x = np.logspace(np.log10(bins_c[0]), np.log10(bins_c[-1]), 1000)
    
    all_normalized_areas = np.concatenate(all_normalized_areas).ravel()
    mean = np.mean(all_normalized_areas)
    std  = np.std(all_normalized_areas)
    a = mean**2 / std**2 
    b = mean / std**2
    rpp = (b**a) / gamma(a) * (x**(a - 1)) * np.exp(-b * x)
    
    a = 3.385 
    b = 3.04011
    c = 1.078
    rpp = c*(b**(a/c))/gamma(a/c)*(x**(a-1))*np.exp(-b*(x**c))
   
    PDF, bins = np.histogram(all_normalized_areas, bins=bins, density=True)    
    idx = np.unravel_index(ii, (2, 2))
    ax[idx].loglog(bins_c, PDF, 'o', color='blue', label='exp. data')
    ax[idx].loglog(x, rpp, '--', color='black', label='RPP')
    ax[idx].set_ylim([1e-4, 1e1])
    ax[idx].legend()
    ax[idx].grid()
    ax[idx].set_title(f'{Alist[ii]}')
    ii += 1

# -----------------------------------------------------------------------------------------
# Plot the average pdfs of all voronoi diagrams for all forcing strenghts on a longer scale
# -----------------------------------------------------------------------------------------
#a, b = 3.45, 3.45
#rpp = (b**a) / gamma(a) * (x**(a-1)) * np.exp(-b * x)
bins   = np.geomspace(1e-2, 10, 50)
bins   = np.linspace(1e-2, 10, 50)
bins_c = (bins[:-1] + bins[1:]) / 2
x      = np.logspace(np.log10(bins_c[0]), np.log10(bins_c[-1]), 100)

# Plot the particle areas 
frames = 3072
all_stds = []
all_means = []
all_partitioned_areas = []

# Create figure 
fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)
ii = 0
for area, Narea, particles in zip(all_areas, area_count, particle_count): 
    # Get the normalized areas 
    stds_per_campaign = []
    stds_per_forcing = []
    means_per_campaign = []
    means_per_forcing = []
    areas_per_campaign = []
    areas_per_forcing = []
    all_normalized_areas = []
    indices = np.concatenate((np.array([0]), np.cumsum(Narea)))
    for i in range(len(indices) - 1):
        A_avg = A_tot / particles[i]
        normalized_area = area[indices[i]:indices[i + 1]] / A_avg
        all_normalized_areas.append(normalized_area)
        stds_per_campaign.append(np.std(normalized_area))
        means_per_campaign.append(np.mean(normalized_area))
        areas_per_campaign.append(normalized_area)
        if (((i % (frames - 1)) == 0) & (i != 0)):
            stds_per_forcing.append(stds_per_campaign)
            stds_per_campaign = []
            means_per_forcing.append(means_per_campaign)
            means_per_campaign = []
            areas_per_forcing.append(areas_per_campaign)
            areas_per_campaign = []
   
    all_partitioned_areas.append(areas_per_forcing)
    all_stds.append(stds_per_forcing)
    all_means.append(means_per_forcing)
    all_normalized_areas = np.concatenate(all_normalized_areas).ravel()
    PDF, _ = np.histogram(all_normalized_areas, bins=bins, density=True)

    idx = np.unravel_index(ii, (2, 2))
    ax = axes[idx]

    mean = np.mean(all_normalized_areas)
    std  = np.std(all_normalized_areas)
    a = mean**2 / std**2 
    b = mean / std**2
    #rpp = rpp(x, a, b) 
    
    a = 3.385 
    b = 3.04011
    c = 1.078
    rpp = rpp_c(x, a, b, c) 
   

    ax.semilogy(bins[:-1], PDF, 'x', color='blue')
    ax.semilogy(x, rpp, '--', color='black')
    ax.set_ylim([1e-5, 1e1])
    ax.set_title(f'{Alist[ii]}')
    ax.set_xlabel(r'$S / \langle S \rangle$')
    ax.set_ylabel(r'$PDF$')
    ax.grid()
    ii += 1

#%% Plot of the change of the standard deviation the measurement time
# Plot the standard deviation over time
n = 50
interval = 500
kk = 0
bins = np.geomspace(1e-2, 15, n)
bins_c = (bins[1:] + bins[:-1]) / 2
indices = np.arange(0, frames)[::interval]
amps = [5, 10, 15, 20]

fps = 60
conv_factor = 5 / 4

fig1, ax1 = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
fig2, ax2 = plt.subplots()
w = np.linspace(0.5, 1, len(indices))

# Every forcing strength
ii = 0
for mean, std, area in zip(all_means, all_stds, all_partitioned_areas):  
    all_a_means = np.zeros(len(indices))
    all_a_stds = np.zeros(len(indices))
    
    # Every campaign
    all_pdfs = np.zeros((len(area), len(indices), len(bins_c)))
    for m, s, ar in zip(mean, std, area):
        a_means = []
        a_stds = []
        prev = 0
        a = np.array(m)**2 / np.array(s)**2
        for ii, idx in enumerate(indices):
            a_means.append(np.mean(s[prev:idx]))
            a_stds.append(np.std(s[prev:idx]))
            for jj in range(prev, idx):
                all_pdfs[kk, ii] += np.histogram(ar[jj], bins=bins, density=True)[0]
            prev = idx
        

    idx = np.unravel_index(ii, (2, 2))
    ax[idx].errorbar(np.arange(0, len(a_means)) * interval, a_means, yerr=a_stds/np.sqrt(len(a_stds)), color='green', ls='--', capsize=5, capthick=1, ecolor='black')
    #ax[idx].plot(a, color='red', ls='--')
    ax[idx].grid()
    #ax[idx].set_ylim([0, 60])
    ax[idx].set_title(f'{Alist[ii]}')
    ax[idx].set_xlabel(r'$frame$')
    ax[idx].set_ylabel(r'$\langle std_{s} \rangle$')
    ii += 1

    axis_idx = np.unravel_index(kk, (2, 2))
    ll = 0
    means = np.zeros(all_pdfs.shape[1])
    for pdf in np.mean(all_pdfs, axis=0):
        pdf = savgol_filter(pdf, 3, 1) 
        idx_max = np.argmax(pdf)
        means[ll] = pdf[idx_max]
        ax1[axis_idx].loglog(bins_c, pdf / interval, '.-', color=lighten_color('b', w[ll]))
        ax1[axis_idx].loglog(x, rpp, '--', color='black')
        ax1[axis_idx].loglog(bins[idx_max], pdf[idx_max] / interval, 'o', color='black')
        ll += 1
    ax1[axis_idx].grid()
    ax1[axis_idx].legend()
    ax1[axis_idx].set_ylim([1e-3, 1e1])
    xx = indices
    ax2.plot(xx, means, 'o--', color=lighten_color('blue', w[kk]))
    kk += 1
ax2.grid()

# Plot the amount of average number of particles/areas in the system
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
print(area_count)
ii = 0
for a in area_count:
    a_stds = []
    a_means = []
    prev = 0
    indices = np.arange(0, frames)[::interval]
    for idx in indices:
        a_means.append(np.mean(a[prev:idx]))
        a_stds.append(np.std(a[prev:idx]))
        prev = idx

    idx = np.unravel_index(ii, (2, 2))
    ax[idx].errorbar(np.arange(0, len(a_means)) * interval, a_means, yerr=a_stds/np.sqrt(len(a_stds)), color='green', ls='--', capsize=5, capthick=1, ecolor='black')
    #ax[idx].plot(a, color='red', ls='--')
    ax[idx].grid()
    #ax[idx].set_ylim([0, 60])
    ax[idx].set_title(f'{Alist[ii]}')
    ax[idx].set_xlabel(r'$frame$')
    ax[idx].set_ylabel(r'$\langle N_{s} \rangle$')
    ii += 1

plt.show()

