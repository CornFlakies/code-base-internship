import argparse
from processdata import HelperFunctions, ProcessData
import matplotlib.pyplot as plt 
import numpy as np

if __name__ == '__main__':
    '''
    If loading in from spyder, comment out or remove the argparser stuff,
    and write down the input_folder and output_folder in this script.

    Then replace those strings with the args.input_folder and args.output_folder
    arguments in the ProcessData calls
    '''
    #Chunks hardcoded at 4
    chunks = 4

    # Load in image paths
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('input_folder', type=str)
    args = argparser.parse_args()

    image_paths, _ = HelperFunctions.load_images(args.input_folder)
    
    plt.figure()
    plt.imshow((np.load(image_paths[0])))
    plt.figure()
    plt.imshow((np.load(image_paths[int(len(image_paths)*.5)])))
    plt.figure()
    plt.imshow((np.load(image_paths[int(len(image_paths)*.75)])))
    plt.figure()
    plt.imshow((np.load(image_paths[-1])))
    plt.show()
