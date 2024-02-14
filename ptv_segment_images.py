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

    # Load in image paths
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('template', type=str)
    argparser.add_argument('output_folder', type=str)
    argparser.add_argument('background_folder', type=str)
    argparser.add_argument('reference_folder', type=str)
    argparser.add_argument('input_folder', type=str)
    args = argparser.parse_args()

    HelperFunctions.create_output_folder(args.output_folder)
    template = sk.io.imread(args.template)

    # Creating a background image by averaging a few frames from the background
    print('generating background image ...')
    image_paths, _ = HelperFunctions.load_images(args.background_folder)
    background_img = np.zeros(np.load(image_paths[0]).shape)
    for image in image_paths:
        background_img += np.load(image)
    background_img /= len(image_paths)

    # Creating a background image by averaging a few frames from the background
    print('generating reference image ...')
    image_paths, _ = HelperFunctions.load_images(args.reference_folder)
    reference_img = np.zeros(np.load(image_paths[0]).shape)
    for image in image_paths:
        reference_img += np.load(image)
    reference_img /= len(image_paths)
    reference_img -= background_img
   
    # Get idx of the carrier peak
    center_line = reference_img
    fft_center_line = np.zeros(np.size(center_line, axis=1) // 2 + 1)
    i = 0 
    for ctr in center_line:
        fft_center_line += np.real(np.fft.rfft(ctr))
        i += 1
    fft_center_line /= i
    ctr_idx = np.argmax(fft_center_line[9:]) + 9 

    # Load the particle images
    image_paths, _ = HelperFunctions.load_images(args.input_folder)
    filename = 'particle_'

    for i, image in enumerate(image_paths[90:]):
        img = np.load(image).astype(np.float64)
        raw_img = img
        img -= background_img
       
        # Generate fft of the image, and center it with fftshift
        ft = np.fft.fft2(img)
        ft = np.fft.fftshift(ft)
    
        # Find the center of the image
        row, col = ft.shape
        row_half = row // 2
        col_half = col // 2
    
        # Filter the carrier peaks using an inverted hann window
        size = 40
        window = np.abs(np.outer(hann(size * 4), hann(size * 4)) - 1)[size:-size, size:-size]
        ft[row_half-size:row_half+size, col_half+ctr_idx-size:col_half+ctr_idx+size] *= window
        ft[row_half-size:row_half+size, col_half-ctr_idx-size:col_half-ctr_idx+size] *= window
    
        # Transform the image back to real space
        img_filtered = np.real(np.fft.ifft2(np.fft.fftshift(ft)))
        shape = img_filtered.shape

        # Match the template to find the particles
        result = sk.feature.match_template(img_filtered, template, pad_input=True)
        
        # Apply a thresholding and invert the image
        idx = result > (result.max() * 0.7)
        result = ((1 - np.ceil(result * idx)) * 255).astype(np.uint8)
       
        # Prepare file for saving to tiff
        name = filename + str(i + 1).zfill(5) + '.tiff'
        file = os.path.join(args.output_folder, name)
        sk.io.imsave(file, result, check_contrast=False)

        if (i % int(len(image_paths) / 10) == 0):
            HelperFunctions.print(f'  {int(np.ceil(i / len(image_paths) * 100))}% ...', mode='o')

