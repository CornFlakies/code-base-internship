import os
import argparse
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
from helper_functions import HelperFunctions as hp

if __name__ == '__main__':
    '''
    If loading in from spyder, comment out or remove the argparser stuff,
    and write down the input_folder and output_folder in this script.

    Then replace those strings with the args.input_folder and args.output_folder
    arguments in the ProcessData calls
    '''

    # Load in image paths
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('output_folder', type=str)
    argparser.add_argument('input_folder', type=str)
    args = argparser.parse_args()

    file_paths, _ = hp.load_images(args.input_folder, header='tif')
    hp.create_output_folder(args.output_folder)

    i = 0
    rows, cols = sk.io.imread(file_paths[0]).shape
    img = np.zeros((rows, cols), dtype=float)
    for file in file_paths:
        img += sk.io.imread(file).astype(np.float64)
        i += 1
    img /= i

    plt.figure()
    plt.imshow(img.astype(np.uint16))
    plt.show()

    filename = 'fondo.npz'
    file = os.path.join(args.output_folder, filename)
    np.savez(file, img.astype(np.uint16))
