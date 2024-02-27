import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# Load in image paths
argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('input_file', type=str)
args = argparser.parse_args()

df = pd.read_pickle(args.input_file)
Ntray = int(df.shape[1]//2)

#plt.title(‘Num. tray = {}‘.format(Ntray))
for n in range(Ntray):
    x = df.iloc[:, 2*n  ].values.astype(float)
    y = df.iloc[:, 2*n+1].values.astype(float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    plt.plot(x, y,'.-', markersize=2)
    plt.plot(x[0], y[0], 'o', markersize=4, color='r', fillstyle=None)
    plt.gca().invert_yaxis()
    #plt.plot(x, y, ‘.-’, markersize=2, color=‘C{}’.format(m%9))
    plt.grid(True)
plt.show()
