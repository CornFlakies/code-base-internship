import argparse
from helper_functions import HelperFunctions as hp

import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation

if __name__ == '__main__':
    '''
    If loading in from spyder, comment out or remove the argparser stuff,
    and write down the input_folder and output_folder in this script.

    Then replace those strings with the args.input_folder and args.output_folder
    arguments in the ProcessData calls
    '''
    # Load in image paths
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('input_folder', type=str)
    args = argparser.parse_args()

    image_paths, _ = hp.load_images(args.input_folder)
    padding = 5 #px

    # Frame data
    max_frame = 300
    print(f'processing the first {max_frame} frames ...')
    frames = len(image_paths[:max_frame])
    fps = 30

    def update_plot(frame_number, image_paths, plot):
        hp.print(f'processing frame {frame_number}', 'o')
        img = np.load(image_paths[frame_number])[padding:-padding, padding:-padding]
        plot[0].remove()
        plot[0] = ax.plot_surface(x, y, img - np.mean(img), cmap='magma')

    # Load in first image
    img = np.load(image_paths[0])[padding:-padding, padding:-padding]
    length = np.size(img, axis=0)
    pix = np.arange(0, length, 1)
    x, y = np.meshgrid(pix, pix)

    # Create first surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot = [ax.plot_surface(x, y, img - np.mean(img), color='0.75', rstride=1, cstride=1, cmap='magma')]
    ax.set_zlim([-.75, .75])
    ax.set_xlabel('length (px)')
    ax.set_ylabel('width (px)')
    ax.set_zlabel('$h_0$ (cm)')

    print('running animation ...')
    ani = animation.FuncAnimation(fig, update_plot, frames, fargs=(image_paths, plot), interval=1000/fps)
    print('done ..')

    fn = 'surface_plot_animation'
    ani.save(fn+'.mp4', writer='ffmpeg', fps=fps)
    #ani.save(fn+'.gif', writer='ffmpeg', fps=fps)
