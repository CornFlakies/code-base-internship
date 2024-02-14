import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import os
import skimage as sk
import argparse

# Load in image paths
argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('input_folder', type=str)
args = argparser.parse_args()
images = np.sort(os.listdir(args.input_folder))

image_paths = []
for entry in images:
    split_filename = entry.split('.')
    if (split_filename[-1] == 'tiff'):
        image_paths.append(os.path.join(args.input_folder, entry))
print(f'loaded {len(image_paths)} images ...')
# Initialize variables to keep track of the current image index
current_image_index = 0

# Create a function to update the displayed image
def update_image(event):
    global current_image_index
    current_image_index = (current_image_index + 100) % len(image_paths)
    img = sk.io.imread(image_paths[current_image_index])
    ax.imshow(img)
    ax.set_title(f'frame {current_image_index}')
    plt.draw()

# Create a matplotlib figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

# Load and display the first image
img = sk.io.imread(image_paths[current_image_index])
ax.imshow(img)

# Create a button widget to switch to the next image
next_button = Button(plt.axes([0.7, 0.05, 0.2, 0.075]), 'Next Image')
next_button.on_clicked(update_image)

plt.show()
