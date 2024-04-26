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

images = np.sort(os.listdir(args.input_folder_rawimg))
df = pd.read_pickle(args.input_file_df)
Ntray = int(df.shape[1]//2)

image_paths = []
for entry in images:
    split_filename = entry.split('.')
    if ((split_filename[-1] == 'tiff') | (split_filename[-1] == 'tif')):
        image_paths.append(os.path.join(args.input_folder_rawimg, entry))
print(f'loaded {len(image_paths)} images ...')
# Initialize variables to keep track of the current image index
current_image_index = 0

# Create a function to update the displayed image
def update_image(event):
    global current_image_index
    current_image_index = (current_image_index + 100) % len(image_paths)
    img = sk.io.imread(image_paths[current_image_index])
    ax.cla()
    ax.imshow(img)
    ax.set_title(f'frame {current_image_index}')

    ax.plot(x[current_image_index], y[current_image_index], '.', markersize=1, color='r', fillstyle=None)

    print(x[0].shape)
    
    plt.draw()

# Create a matplotlib figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

# Load and display the first image
img = sk.io.imread(image_paths[current_image_index])


xidx = np.arange(0, df.shape[1] - 1, 2)
yidx = np.arange(1, df.shape[1], 2)
x = df.iloc[:, xidx].values.astype(float)
y = df.iloc[:, yidx].values.astype(float)

print(len(xidx))

ax.plot(x[0], y[0], 'o', markersize=1, color='r', fillstyle=None)
ax.imshow(img)

# Create a button widget to switch to the next image
next_button = Button(plt.axes([0.7, 0.05, 0.2, 0.075]), 'Next Image')
next_button.on_clicked(update_image)

plt.show()
