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
argparser.add_argument('input_dir', type=str)
args = argparser.parse_args()

file_paths, _ = hp.load_images(args.input_dir, header='out')

bins= 80
fps = 60
i=0
totcountx=np.zeros(bins)
totcounty=totcountx.copy()
totcountax=totcountx.copy()
totcountay=totcountx.copy()
points = np.linspace(-4, 4, 100)

def gaussian(x, sigma = 1, mu = 0):
    return  np.exp(- 0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

all_vx = 0
all_vy = 0
for file in file_paths:
    df = pd.read_pickle(file)
    Ntray = int(df.shape[1]//2)
    conv_factor = 0.027 #cm / px
    for n in range(Ntray):
        x = df.iloc[:, 2*n  ].values.astype(float) * conv_factor
        y = df.iloc[:, 2*n+1].values.astype(float) * conv_factor
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        
        vx = np.diff(x) 
        vy = np.diff(y) 
        ax = np.diff(x, n=2)
        ay = np.diff(y, n=2)
        
        countx, binsx = np.histogram((vx - vx.mean()) / vx.std(), bins=bins, range=[-4, 4], density=True)
        county, binsy = np.histogram((vy - vy.mean()) / vy.std(), bins=bins, range=[-4, 4], density=True)
        countax, binsax = np.histogram((ax - ax.mean()) / ax.std(), bins=bins, range=[-4, 4], density=True)
        countay, binsay = np.histogram((ay - ay.mean()) / ay.std(), bins=bins, range=[-4, 4], density=True)

        all_vx += vx.std()
        all_vy += vy.std()

        totcountx += countx
        totcounty += county
        totcountax += countax
        totcountay += countay
        i += 1 

    all_vx /= n 
    all_vy /= n

print(all_vx * fps)
print(all_vy * fps) 

plt.figure()
plt.semilogy(binsx[:-1], totcountx/i, '.', label=r"$v_x$")
plt.semilogy(binsy[:-1], totcounty/i, '.', label=r"$v_y$")
plt.semilogy(points, gaussian(points), '--', color='black', label='$N(0,1)$')
plt.xlabel(r"$(v - \langle v \rangle) / \sigma_{v}$")
plt.ylabel(r"$PDF_{v}$")
plt.legend()
plt.grid()

plt.figure()
plt.semilogy(binsax[:-1], totcountax/i, '.', label=r"$a_x$")
plt.semilogy(binsay[:-1], totcountay/i, '.', label=r"$a_y$")
plt.semilogy(points, gaussian(points), '--', color='black', label=r'$N(0,1)$')
plt.xlabel(r"$(a - \langle a \rangle) / \sigma_{a}$")
plt.ylabel(r"$PDF_{a}$")
plt.legend()
plt.grid()
plt.show()
