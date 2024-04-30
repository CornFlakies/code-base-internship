# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.close('all')
class ParticleDispersion:
    def __init__(self, N, box_size, frames, dt, sigma, tol, preCalcData=False):
        self.tol = tol
        self.box_size = box_size
        self.dt = dt
        self.sigma = sigma
        self.frames = frames
        
        if not preCalcData:
            self.N = N
            self.Npositions = np.zeros((frames, N, 2))
            self.Npositions[0] = np.random.uniform(0, 1, size=(N, 2)) * self.box_size
       
        else:
            self.propagate_particles = self.doNone
            _, self.N, _ = N.shape 
            self.Npositions = N

        # Define particle indices
        self.xidx, self.yidx = np.meshgrid(np.arange(0, self.N), np.arange(0, self.N))
   
    def doNone(self, i):
        return None

    # Find interparticle dinstances
    def get_interparticle_distances(self, pos):
        # Compute interparticle distances
        diff_pos = pos[self.yidx] - pos[self.xidx]

        # Get the interparticle distances, find the linked particles
        dij = np.linalg.norm(diff_pos, axis=-1)       
        
        return dij

    # Find particles pairs before the start of the measurement
    def get_particle_pairs(self):
        # Get current interparticle distances
        dij = self.get_interparticle_distances(self.Npositions[0])
        dij = np.where(dij < self.tol, 1, 0)        
        
        # Set the diagonal to 0, and ignore the lower triangular, to prevent duplication of distances
        dij[np.arange(0, self.N), np.arange(0, self.N)] = 0
        dij[np.tril_indices(np.size(dij, axis=0))] = 0

        # Create linked list and initial distances list
        linked_list = []
        pair_count = 0
        for row in dij:
            indices = []
            for val in np.argwhere(row == 1):        
                indices.append(*val)
                pair_count += 1
            linked_list.append(indices)
                
        self.deltaR = np.zeros((pair_count, self.frames))
        self.linked_list = linked_list
        self.comp_pair_distance(0)
    
    # Propagate particles in time, if doing a random simulation
    def propagate_particles(self, i):
        velocities = np.random.normal(0, self.sigma, size=(self.N, 2))
        
        self.Npositions[i] += self.Npositions[i - 1] + velocities * self.dt
        self.Npositions[i] = np.where(self.Npositions[i] > self.box_size, 
                                      self.Npositions[i - 1] - velocities * self.dt, 
                                      self.Npositions[i])
        self.Npositions[i] = np.where(self.Npositions[i] < 0, 
                                      self.Npositions[i - 1] - velocities * self.dt, 
                                      self.Npositions[i])
        
        #self.Npositions[i, np.argwhere(np.abs(self.Npositions[i]) > self.box_size)] //= self.box_size
    
    # Compute the mean particle pair distance per frame
    def comp_pair_distance(self, frame):
        # Get interparticle distances
        dij = self.get_interparticle_distances(self.Npositions[frame])
        
        # Pair distance
        j = 0
        for i, pairs in enumerate(self.linked_list):
            if pairs:
                for pair in pairs:
                    self.deltaR[j, frame] = dij[i, pair]
                    j += 1

    # Run pair dispersion algorithm
    def run(self):
        self.get_particle_pairs()
        
        for i in tqdm(range(1, self.frames)):
            self.propagate_particles(i)
            self.comp_pair_distance(i)
        
        for i in range(self.deltaR.shape[0]):
            self.deltaR[:, i] /= self.deltaR[:, 0]

    # Getter function for the computed data
    def get_data(self):
        return self.linked_list, self.Npositions, self.deltaR
