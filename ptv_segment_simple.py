import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import cv2 as cv
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

    # Load in image paths
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('template', type=str)
    argparser.add_argument('output_folder', type=str)
    argparser.add_argument('background_file', type=str)
    argparser.add_argument('input_folder', type=str)
    args = argparser.parse_args()

    HelperFunctions.create_output_folder(args.output_folder)
    
    template = sk.io.imread(args.template)
    background = sk.io.imread(args.background_file)

    # Load the particle images
    image_paths, _ = HelperFunctions.load_images(args.input_folder, header='tif')
    filename = 'particle_'

    for i, image in enumerate(image_paths):
        img = sk.io.imread(image).astype(np.float64)
        raw_img = img
      
        ##temp
        #temp = img[280:300, 602:622]
        #sk.io.imsave('template.tiff', temp)
        #plt.figure()
        #plt.imshow(temp)
 
        # Match the template to find the particles
        result = sk.feature.match_template(img, template, pad_input=True)        

       # plt.figure()
       # plt.imshow(result)
         
        # Apply a thresholding and invert the image
        idx = result > (result.max() * 0.85)
        result = ((1 - np.ceil(result * idx)) * 255).astype(np.uint8)

        # plt.figure()
        # plt.imshow(img)
        # plt.figure()
        # plt.imshow(result)
        # plt.show()

        # Prepare file for saving to tiff
        name = filename + str(i + 1).zfill(5) + '.tiff'
        file = os.path.join(args.output_folder, name)
        sk.io.imsave(file, result, check_contrast=False)

        if (i % int(len(image_paths) / 10) == 0):
            HelperFunctions.print(f'  {int(np.ceil(i / len(image_paths) * 100))}% ...', mode='o')

