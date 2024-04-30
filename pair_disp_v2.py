# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from pandas.core.frame import NaPosition
from tqdm import tqdm
import pandas as pd
import numpy as np

plt.close('all')
class ParticleDispersion:
    def __init__(self, df, box_size, tol):
        # Define analysis parameters
        self.tol = tol
        self.box_size = box_size
       
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
            self.Npositions[:, n, 0] = x
            self.Npositions[:, n, 1] = y

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
        for row in dij:
            indices = []
            for val in np.argwhere(row == 1):        
                indices.append(*val)
            linked_list.append(indices) 
        return linked_list

    def remove_and_add_pairs(self, dij, curr_list):
        for ii, row in enumerate(dij):
            currentPairs = curr_list[ii]
            for idxPair in np.argwhere(row == 1):
                if idxPair in currentPairs:
                    print(idxPair)
        return None

    # Find particles pairs before the start of the measurement
    def get_particle_pairs(self, i, curr_list=None):
        # Get current interparticle distances
        dij = self.get_interparticle_distances(self.Npositions[i])
        dij = np.where(dij > self.tol, 0, dij)
        dij = np.where((dij <= self.tol) & (dij != 0), 1, dij)
        
        # Set the diagonal to 0, and ignore the lower triangular, to prevent duplication of distances
        dij[np.arange(0, self.N), np.arange(0, self.N)] = 0
        dij[np.tril_indices(np.size(dij, axis=0))] = 0 
       
        if (curr_list is None):
            linked_list = self.init_link_list(dij)
            deltaR = np.zeros((len(linked_list), self.frames))
        else:
            linked_list = self.remove_and_add_pairs(dij, curr_list)
            deltaR = None

        return linked_list, deltaR

    # Compute the pair distance per frame
    def comp_pair_distance(self, frame, linked_list):
        # Get interparticle distances
        dij = self.get_interparticle_distances(self.Npositions[frame])
        
        # Pair distance
        j = 0
        for i, pairs in enumerate(linked_list):
            if pairs:
                for pair in pairs:
                    self.deltaR[j, frame] = dij[i, pair]
                    j += 1

    # Run pair dispersion algorithm
    def run(self):
        linked_list, self.deltaR = self.get_particle_pairs(0, curr_list=None)
        # TODO Compute initial distances 
        for i in tqdm(range(self.frames)):
            linked_list = self.get_particle_pairs(i, linked_list)
            self.comp_pair_distance(i, linked_list)

        for i in range(1, self.deltaR.shape[1]):
            self.deltaR[:, i] -= self.deltaR[:, 0]
        
        return self.deltaR, linked_list
