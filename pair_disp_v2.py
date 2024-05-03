# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from pandas.core.frame import NaPosition
from tqdm import tqdm
import pandas as pd
import numpy as np

plt.close('all')
class ParticleDispersion:
    def __init__(self, df, tol, tol_thickness, minSize, maxSize):
        # Define analysis parameters
        self.tolmin = tol - tol_thickness
        self.tolmax = tol + tol_thickness

        # Get particle array shape
        df = pd.read_pickle(df)
        self.N = df.shape[1] // 2
        self.frames = df.shape[0]
        self.Npositions = np.zeros((self.frames, self.N, 2))

        # Define global linked particle array
        self.Npairs = np.array([])

        # Get particle x and y position
        for n in range(self.N):     
            x = df.iloc[:, 2*n  ].values.astype(float)
            y = df.iloc[:, 2*n+1].values.astype(float)
            
            idx = []
            idx.append(np.argwhere(x < minSize))
            idx.append(np.argwhere(y < minSize))
            idx.append(np.argwhere(x > maxSize))
            idx.append(np.argwhere(y > maxSize))
            if idx:
                idx = np.concatenate(idx).ravel()
                x[idx] = np.nan 
                y[idx] = np.nan

            self.Npositions[:, n, 0] = x
            self.Npositions[:, n, 1] = y

            # Filter 
            idx = np.argwhere(np.abs(np.diff(np.linalg.norm(self.Npositions[:, n], axis=-1)) > 6))
            if (idx.size != 0):
                cutoff = idx[0][0]
                self.Npositions[cutoff:, n] = np.nan
        
        # Define particle indices
        self.xidx, self.yidx = np.meshgrid(np.arange(0, self.N), np.arange(0, self.N))
  
    # Find interparticle dinstances
    def get_interparticle_distances(self, pos):
        # Compute interparticle distances
        diff_pos = pos[self.yidx] - pos[self.xidx]

        # Get the interparticle distances, find the linked particles
        dij = np.linalg.norm(diff_pos, axis=-1)  
        return dij

    def init_link_list(self, dij):
        # Create initial linked list
        linked_list = []
        pair_count = 0
        for row in dij:
            indices = []
            for val in np.argwhere(row == 1):        
                indices.append(*val)
                pair_count += 1
            linked_list.append(indices) 
        deltaR = np.ones((pair_count, self.frames)) * np.nan
        pair_indices = np.arange(0, pair_count, 1)

        return linked_list, deltaR, pair_indices

    def remove_and_add_pairs(self, dij, curr_deltaR, curr_list, pair_indices): 
        current_pair_index = 0
        for ii, curr_list_row in enumerate(curr_list):
            for curr_list_elem in curr_list_row:
                # Skip empty entries
                if not curr_list_elem:
                    break
                # Remove particle from tracker if nan is found
                elif np.isnan(dij[ii, curr_list_elem]):
                    pair_indices = np.delete(pair_indices, current_pair_index)
                    curr_list[ii].remove(curr_list_elem)        
                    current_pair_index += 1
                # Else do nothing and propagate further
                else:
                    current_pair_index += 1

        return curr_list, curr_deltaR, pair_indices

    # Find particles pairs before the start of the measurement
    def get_particle_pairs(self, i, curr_deltaR=None, curr_list=None, pair_indices=None):
        # Get current interparticle distances
        dij = self.get_interparticle_distances(self.Npositions[i])
        dij = np.where(dij > self.tolmax, 0, dij)
        dij = np.where(((dij <= self.tolmax) & (dij >= self.tolmin) & (dij != 0)), 1, dij)
        
        # Set the diagonal to 0, and ignore the lower triangular, to prevent duplication of distances
        dij[np.arange(0, self.N), np.arange(0, self.N)] = 0
        dij[np.tril_indices(np.size(dij, axis=0))] = 0 
       
        if ((curr_list is None) & (curr_deltaR is None)):
            linked_list, deltaR, pair_indices = self.init_link_list(dij)
        else:
            linked_list, deltaR, pair_indices = self.remove_and_add_pairs(dij, curr_deltaR, curr_list, pair_indices)
        
        self.linked_list = linked_list
        self.deltaR = deltaR
        self.pair_indices = pair_indices

    # Compute the pair distance per frame
    def comp_pair_distance(self, frame):
        # Get interparticle distances
        dij = self.get_interparticle_distances(self.Npositions[frame])
        
        # Pair distance
        j = 0
        for i, pairs in enumerate(self.linked_list):
            if pairs:
                for pair in pairs:
                    self.deltaR[self.pair_indices[j], frame] = dij[i, pair]
                    j += 1

    # Run pair dispersion algorithm
    def run(self):
        self.get_particle_pairs(0)
        print(self.deltaR.shape)
        
        # TODO Compute initial distances 
        for i in tqdm(range(self.frames)): 
            try:
                self.get_particle_pairs(i, self.deltaR, self.linked_list, self.pair_indices)
            except:
                break
            self.comp_pair_distance(i)

        return self.deltaR, self.Npositions
