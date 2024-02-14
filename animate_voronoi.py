from helper_functions import HelperFunctions as hp
from scipy.spatial import Voronoi
from scipy.spatial import voronoi_plot_2d

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

if __name__ == "__main__":
    '''
    If loading in from spyder, comment out or remove the argparser stuff,
    and write down the input_folder and output_folder in this script.

    Then replace those strings with the args.input_folder and args.output_folder
    arguments in the ProcessData calls
    '''
    # Load in image paths
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('input_file_DF', type=str)
    args = argparser.parse_args()

    df = pd.read_pickle(args.input_file_DF)
    Ntray = int(df.shape[1]//2)

    max = 0
    for n in range(Ntray):
        x = df.iloc[:, 2*n  ].values.astype(float)
        y = df.iloc[:, 2*n+1].values.astype(float)
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
    
        if (len(x) > max):
            max = len(x)

    def get_points(Ntray, i):
        points = [] 
        for n in range(Ntray):
            x = df.iloc[:, 2*n  ].values.astype(float)
            y = df.iloc[:, 2*n+1].values.astype(float)
            x = x[~np.isnan(x)]
            y = y[~np.isnan(y)]
            
            if not (len(x) <= i):
                points.append([x[i], y[i]])
        return points

    # Frame data
    max_frame = max
    print(f'processing the first {max_frame} frames ...')
    fps = 30

    # Create first surface plot 
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 1024)
    
    def updater(i):
        hp.print(f'processing frame {i}', 'o') 
        ax.cla()
        vor = Voronoi(get_points(Ntray, i))
        voronoi_plot_2d(vor, ax=ax)
    
    print('running animation ...')
    #ani = animation.FuncAnimation(fig, update_plot, max_frame, fargs=(plot), interval=1000/fps)
    ani = animation.FuncAnimation(fig, updater, frames=max_frame, interval=1000/fps)
    print('done ..')

    fn = 'voronoi_animation'
    ani.save(fn+'.mp4', writer='ffmpeg', fps=fps)
    #ani.save(fn+'.gif', writer='ffmpeg', fps=fps)
