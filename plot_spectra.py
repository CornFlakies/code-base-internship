import argparse
from processdata import ProcessData

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
        
    processor = ProcessData(args.input_folder)
    processor.plot_power_spectra()
