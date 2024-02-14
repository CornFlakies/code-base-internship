from helper_functions import HelperFunctions as hp
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import skimage as sk
import pandas as pd
import numpy as np
import argparse
import os

# Load in image paths
argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('input_folder_rawimg', type=str)
argparser.add_argument('input_file_df', type=str)
args = argparser.parse_args()

raw_img_paths, _ = hp.load_images(args.input_folder_rawimg)

df = pd.read_pickle(args.input_file_df)
Ntray = int(df.shape[1]//2)

plt.title("Num. tray = {}".format(Ntray))
for i, img in enumerate(raw_img_paths[90:]):    
    for n in range(Ntray):
        x = df.iloc[:, 2*n  ].values.astype(float)
        y = df.iloc[:, 2*n+1].values.astype(float)
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
    
        plt.plot(x[i], y[i], 'o', markersize=4, color='r', fillstyle=None)
        plt.gca().invert_yaxis()

    image = np.load(img)    
    plt.imshow(image)
    plt.show()

