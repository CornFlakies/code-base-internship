import argparse
from processdata import HelperFunctions, ProcessData
import matplotlib.pyplot as plt
import numpy as np 
import skimage as sk

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

    processor = ProcessData(args.input_folder)
    image_paths, image_names = HelperFunctions.load_images(args.input_folder)

    img = np.load(image_paths[0])
    dot = img[211:231, 351:371]
    res = sk.feature.match_template(img, dot)
    bin = res > (res.max() - .2)


    plt.imshow(bin)
    plt.show()
