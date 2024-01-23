import argparse
from processdata import ProcessData

if __name__ == '__main__':
    '''
    If loading in from spyder, comment out or remove the argparser stuff,
    and write down the input_folder and output_folder in this script.

    Then replace those strings with the args.input_folder and args.output_folder
    arguments in the ProcessData calls
    '''
    #Chunks hardcoded at 4
    chunks = 4
    #h0 = 4
    #box_length = 30

    # Load in image paths
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('input_folder', type=str)
    argparser.add_argument('conver_factor', type=float)
    argparser.add_argument('h0', type=float)
    args = argparser.parse_args()

    conver_factor = args.conver_factor * 1E-2 # m px**(-1)
    h0 = args.h0 * 1E-2 # m
   
    # Plot all chunks
    dataprocessor = ProcessData(args.input_folder)
    dataprocessor.plot_dispersion_rel(h0, conver_factor)
