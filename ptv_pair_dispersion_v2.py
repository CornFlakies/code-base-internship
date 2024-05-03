import argparse
import get_traj
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pair_disp_v2 import ParticleDispersion
from helper_functions import HelperFunctions as hp

font = {'family' : 'normal',
        'size'   : 10}

matplotlib.rc('font', **font)

# Load in image paths
argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('input_dir', type=str)
args = argparser.parse_args()
file_paths, _ = hp.load_images(args.input_dir, header='out')

# Experimental parameters
conv_fact = 0.8 #mm px**-1
eta = 7.11 # mm
fps = 60
tol_arr = [2, 4, 16, 32] 
tol = np.array(tol_arr) * eta / conv_fact
tol_thickness = np.array([1, 5, 5, 5]) * eta / conv_fact

# Hist data
tot_bins = 300
bins = np.geomspace(1e-2, 300, tot_bins)
bins_c = (bins[:-1] + bins[1:]) / 2
offset = 0.6

# Plotting parameters
cutoff = 1500 #px
grad = np.linspace(0.75, 1, 4)
color = ['blue', 'red', 'green', 'orange']

#fig, ax = plt.subplots(nrows=1, ncols=2)
fig2, ax2 = plt.subplots(figsize=(7.5, 7.5))
fig3, ax3 = plt.subplots(figsize=(7.5, 7.5))
for ii, t in enumerate(tol):
    all_deltaR_means = np.ones((len(file_paths), 3072)) * np.nan
    all_positions_means = all_deltaR_means.copy()
    all_deltaR_pdfs = np.zeros((len(file_paths), len(bins_c)))
    for jj, file in enumerate(file_paths):
        image_size = 1024
        
        part_disp = ParticleDispersion(df=file,      
                                       tol=t,
                                       tol_thickness = tol_thickness[ii],
                                       minSize=200,
                                       maxSize=800)
        deltaR, positions = part_disp.run()
        
        deltaR *= conv_fact
        positions *= conv_fact

        # Plot all delta Rs
        #plt.figure()
        #for dr in deltaR:
        #    plt.loglog(dr**2, '.-')

        # Plot the mean of all delta Rs with respect to the initial particle distance 
        for N in range(deltaR.shape[0]):
            deltaR[N, :] -= deltaR[N, 0]  
        means = []
        pdfs = np.zeros(len(bins_c))
        for Npos_per_frame in deltaR.T:
            dr = Npos_per_frame[~np.isnan(Npos_per_frame)]**2
            if dr.size != 0:
                means.append(np.mean(dr))
                PDF, _ = np.histogram(np.sqrt(dr), bins=bins)
                pdfs += PDF
        all_deltaR_pdfs[jj] += pdfs
        all_deltaR_means[jj, :len(means)] = means

        # Plot the mean of all delta Rs with respect to the particles initial position
        for N in range(positions.shape[1]):
            positions[:, N, :] -= positions[0, N, :]
        positions = np.linalg.norm(positions, axis=-1)
        means = [] 
        for Npos_per_frame in positions:
            dr = np.mean(Npos_per_frame[~np.isnan(Npos_per_frame)]**2)
            means.append(dr)
        #plt2.loglog(means, '.-')
        all_positions_means[jj, :len(means)] = means
    time = np.arange(0, 3072) / fps
    means = []
    for Npos_per_frame in all_deltaR_means.T:
        means.append(np.mean(Npos_per_frame[~np.isnan(Npos_per_frame)])) 
    ax2.loglog(time[:-cutoff], means[:-cutoff], '.-', color=color[ii], label=rf'${tol_arr[ii]}\eta$')
    ax2.axvline(x=0.15, linestyle='--', color='grey')
    ax2.axvline(x=0.4, linestyle='--', color='grey')
    all_deltaR_pdf_means = np.mean(all_deltaR_pdfs, axis=0)
    ax3.loglog(bins_c, all_deltaR_pdf_means / max(all_deltaR_pdf_means), '.', color=color[ii], label=rf'${tol_arr[ii]}\eta$')
    ax3.set_ylabel(r'$P(\delta r)$', fontsize=12)
    ax3.set_xlabel(r'$\delta r\.\. [mm]$', fontsize=12)

def power_law(A, x, p):
    return A * (x/x[0])**(p)

ax2.loglog(time[1:3], power_law(1, time[1:3], 2), '-.', color='black', label=r'$t\sim 2$')
ax2.loglog(time[35:150], power_law(50, time[35:150], 1.5), '--', color='black', label=r'$t\sim 1.5$')
ax2.set_xlabel(r'$t\.\. [s]$', fontsize=12)
ax2.set_ylabel(r'$\langle \delta r^2 \rangle\. [mm^2]$', fontsize=12)
ax2.grid()
ax2.legend()

ax3.grid() 

fig2.savefig('./deltaR_10mm.png')
fig3.savefig('./rayleight_10mm.png')
plt.show()


