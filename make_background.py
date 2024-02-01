import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
from scipy.signal.windows import hann
from helper_functions import HelperFunctions

if __name__ == '__main__':
    '''
    If loading in from spyder, comment out or remove the argparser stuff,
    and write down the input_folder and output_folder in this script.

    Then replace those strings with the args.input_folder and args.output_folder
    arguments in the ProcessData calls
    '''
    #Chunks hardcoded at 4
    chunks = 4
    rows = 995
    cols = 995

    # Load in image paths
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('output_folder', type=str)
    args = argparser.parse_args()

    HelperFunctions.create_output_folder(args.output_folder)
    img = (np.ones((rows, cols)) * 255).astype(np.uint8)
    filename = 'fondo.npz'
    file = os.path.join(args.output_folder, filename)
    np.savez(file, img)
