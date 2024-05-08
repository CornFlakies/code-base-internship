import argparse
import get_traj
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pair_disp_v2 import ParticleDispersion
from helper_functions import HelperFunctions as hp

font = {'size'   : 10}
matplotlib.rc('font', **font)

# Load in image paths
argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('input_dir', type=str)
args = argparser.parse_args()
file_paths, _ = hp.load_images(args.input_dir, header='out')
campaign_folders = hp.load_folders(args.input_dir)
print(campaign_folders)

# Experimental parameters
conv_fact = 0.8 #mm px**-1
eta = 7.11 # mm
fps = 60
tol_arr = [1, 4, 16, 32] 
tol = 2 * eta / conv_fact
tol_thickness = 2 * eta / conv_fact
time = np.arange(0, 3072) / fps

# Hist data
tot_bins = 300
bins = np.geomspace(1e-2, 300, tot_bins)
bins_c = (bins[:-1] + bins[1:]) / 2
offset = 0.6

# Plotting parameters
cutoff = 1500 #px
grad = np.linspace(0.75, 1, 4)
color = 'green'

Alabel = ['10mm', '15mm', '5mm', '20mm']

#fig, ax = plt.subplots(nrows=1, ncols=2)
fig1, ax1 = plt.subplots(figsize=(8, 7.5))
fig2, ax2 = plt.subplots(figsize=(8, 7.5))

all_pdfs = np.zeros((len(campaign_folders), len(bins_c)))
all_drs = np.ones((len(campaign_folders), 3072)) * np.nan

for jj, folder in enumerate(campaign_folders):
    files = hp.load_images(folder, header='out')[0]
    campaign_drs = np.ones((len(files), 3072)) * np.nan
    campaign_pdfs = np.zeros((len(files), len(bins_c))) 
    for ii, campaign_file in enumerate(files):
        print(campaign_file)
        image_size = 1024     
        part_disp = ParticleDispersion(df=campaign_file,      
                                       tol=tol,
                                       tol_thickness = tol_thickness,
                                       minSize=200,
                                       maxSize=800)
        deltaR, _ = part_disp.run()
     
        deltaR *= conv_fact

        # Plot the mean of all delta Rs with respect to the initial particle distance 
        for N in range(deltaR.shape[0]):
            deltaR[N, :] -= deltaR[N, 0]   
        
        means = []
        for Npos_per_frame in deltaR.T:
            dr = np.mean(Npos_per_frame[~np.isnan(Npos_per_frame)]**2)
            means.append(dr)
        campaign_drs[ii, :len(means)] = means 

        for pair in deltaR:
            campaign_pdfs[ii] += np.histogram(pair, bins=bins)[0]
   
    means = []
    for Npos_per_frame in campaign_drs.T:
        dr = np.mean(Npos_per_frame[~np.isnan(Npos_per_frame)])
        means.append(dr)
    
    all_pdfs[jj] = np.mean(campaign_pdfs, axis=0)
    pdf = np.mean(campaign_pdfs, axis=0)
    pdf /= max(pdf)

    ax1.loglog(time[:-1500], means[:-1500], '.-', label=f'A = {Alabel[jj]}')
    ax2.loglog(bins_c, pdf, '.', label=f'A = {Alabel[jj]}') 

ax1.set_xlabel(r'$t\.\. [s]$', fontsize=12)
ax1.set_ylabel(r'$\langle \delta r^2 \rangle\. [mm^2]$', fontsize=12)

ax2.set_ylabel(r'$P(\delta r)$', fontsize=12)
ax2.set_xlabel(r'$\delta r\.\. [mm]$', fontsize=12)
    
ax1.grid()
ax2.grid()

ax1.legend()

plt.show()
