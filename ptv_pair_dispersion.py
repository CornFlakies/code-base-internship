import argparse
import get_traj
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import HelperFunctions as hp
from particle_dispersion import ParticleDispersion

# Load in image paths
argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('input_dir', type=str)
args = argparser.parse_args()
file_paths, _ = hp.load_images(args.input_dir, header='out')

conv_fact = 5/4 #mm px**-1
fps = 60
tol = [200]

# Hist data
tot_bins = 100
bins = np.geomspace(1, 30, tot_bins)
bins_c = (bins[:-1] + bins[1:]) / 2

fig, ax = plt.subplots(nrows=1, ncols=2)
for ii, t in enumerate(tol):
    all_positions = []
    all_deltaR = []
    for file in file_paths:
        xP, yP = get_traj.get_trajectories(file,
                                       minSize=100,
                                       maxSize=800)
        positions = np.array([xP, yP]).T
        image_size = 1024
        frames = positions.shape[0]
        dt = 1 
        sigma = 1.05 
        tol = t
        
        part_disp = ParticleDispersion(N=positions, 
                                       box_size=image_size, 
                                       frames=frames, 
                                       dt=1, 
                                       sigma=1.05, 
                                       tol=tol, 
                                       preCalcData=True)
        part_disp.run()
        
        linked_list, _, deltaR = part_disp.get_data()
      
#        plt.figure()
#        for x, y in zip(xP, yP):
#            plt.plot(x, y, '.-')
#
#        plt.figure()
#        for dr in deltaR:
#            plt.loglog(dr**2, '.-')
#        plt.grid()

        all_positions.append(positions)
        all_deltaR.append(deltaR) 

    minLen = frames
    for nn, dr in enumerate(all_deltaR):
        if (dr.shape[1] < minLen):
            minLen = dr.shape[1]

    all_drs = np.zeros((nn + 1, minLen - 1))
    all_drs_hists = []
    all_dpos = np.zeros((nn + 1, minLen - 1))
    all_dpos_hists = []
    pair_count = []
    for pos, dr in zip(all_positions, all_deltaR): 
        # Plot the change in interparticle distance over time
        pair_count.append(np.size(dr, axis=0))

        pos = np.linalg.norm(pos, axis=-1)
        print(pos.shape)
        for i in range(1, pos.shape[0]):
            pos[i, :] -= pos[0, :]

        # dr of the particle pairs
        pairs = all_deltaR[ii].shape[0]
        all_drs[ii] = np.mean(dr[:, 1:minLen]**2, axis=0)
        all_drs_hists.append(dr[:, 1:].ravel())

        # Compute of trajectories with themselves
        all_dpos[ii] = np.mean(np.abs(pos[1:minLen])**2, axis=-1)
        all_dpos_hists.append(pos[1:].ravel())

        # Plot the individual trajectories, of every campaign
        ax[0].loglog((np.arange(0, 3072, 1) * 1/fps)[1:minLen], all_drs[ii], '.-', label=f'{ii} nr. pairs: {pairs + 1}')
        ax[0].axvline(x=1/4, linestyle='--', color='black')
        ax[0].set_xlabel('t (s)')
        ax[0].set_ylabel(r'$\langle\delta r [mm^{2}]\rangle$')
        #ax[0].set_xlim([0, 60])
        #ax[0].set_ylim([1e1, 1e3])
        ax[0].legend()
        ax[0].grid() 
        
        ax[1].loglog((np.arange(0, 3072, 1) * 1/fps)[1:minLen], all_dpos[ii], '.-')
        ax[1].axvline(x=1/4, linestyle='--', color='black')
        ax[1].set_xlabel('t (s)')
        ax[1].set_ylabel(r'$\langle\delta r [mm^{2}]\rangle$')
        #ax[1].set_xlim([0, 60])
        #ax[1].set_ylim([1e3, 1e3])
        ax[1].grid() 
        ii += 1

# Plot the average over all trajectories
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].loglog((np.arange(0, 3072, 1) * 1/fps)[1:minLen], np.mean(all_drs, axis=0), '.-')
ax[0].axvline(x=1/4, linestyle='--', color='black')
ax[0].set_xlabel('t (s)')
ax[0].set_ylabel(r'$\langle\delta r [mm^{2}]\rangle$')
#ax[0].set_xlim([0, 60])
#ax[0].set_ylim([1e1, 1e3])
ax[0].legend()
ax[0].grid() 

ax[1].loglog((np.arange(0, 3072, 1) * 1/fps)[1:minLen], np.mean(all_dpos, axis=0), '.-')
ax[1].axvline(x=1/4, linestyle='--', color='black')
ax[1].set_xlabel('t (s)')
ax[1].set_ylabel(r'$\langle\delta r [mm^{2}]\rangle$')
#ax[1].set_xlim([0, 60])
#ax[1].set_ylim([1e3, 1e3])
ax[1].grid()


idx = 2

#all_dpos_hists = np.concatenate(all_dpos_hists)
dpos_counts, _ = np.histogram(all_dpos_hists[idx], bins=bins, density=True)

#all_drs_hists = np.concatenate(all_drs_hists)
drs_counts, _ = np.histogram(all_drs_hists[idx], bins=bins, density=True)

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].semilogy(bins_c, drs_counts, '.', color='blue')
ax[0].grid()

ax[1].semilogy(bins_c, dpos_counts, '.', color='blue')
ax[1].grid()

plt.show()


# Plot the particle displacement with itself
#plt.figure()
#for pos, dr in zip(all_positions, all_deltaR):
#    print(pos.shape)
#    deltaPos = pos.copy()
#    for i in range(pos.shape[1]):
#        deltaPos[:, i] -= deltaPos[0, i]
#        deltaPos = np.linalg.norm(deltaPos, axis=-1)
#        deltaPosSq = np.mean(deltaPos**2, axis=-1)
#    print(deltaPosSq.shape)
#    plt.loglog(deltaPosSq)
#plt.show()
