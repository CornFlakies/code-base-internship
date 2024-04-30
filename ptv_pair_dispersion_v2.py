import argparse
import get_traj
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import HelperFunctions as hp
from pair_disp_v2 import ParticleDispersion

# Load in image paths
argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('input_dir', type=str)
args = argparser.parse_args()
file_paths, _ = hp.load_images(args.input_dir, header='out')

conv_fact = 5/4 #mm px**-1
fps = 60
tol = [20]

# Hist data
tot_bins = 100
bins = np.geomspace(1, 30, tot_bins)
bins_c = (bins[:-1] + bins[1:]) / 2

fig, ax = plt.subplots(nrows=1, ncols=2)
for ii, t in enumerate(tol):
    all_positions = []
    all_deltaR = []
    for file in file_paths:
        image_size = 1024
        tol = t
        
        part_disp = ParticleDispersion(df=file, 
                                       box_size=image_size, 
                                       tol=tol)
        part_disp.run()
        
        linked_list, deltaR = part_disp.get_data()
      
