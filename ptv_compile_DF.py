import argparse
import glob, os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt 
from os.path import dirname, join
from helper_functions import HelperFunctions as hp

#--------------------------------------------
def generar_dataframe_filtrado(folderDFs, length_threshold=10, print_progress = True):
        """
        Genera un gran dataframe filtrado en longitud de trayectorias.
        """
    
        # read DFs and IMs file list
        files_DFs = sorted(glob.glob(folderDFs + 'DF_X_*_*_*.out'))
        ID = files_DFs[0].split('/')[-1][:-11]

        # pandas append and/or concat are slow 
        # so we create the output DF once 
    
        # este barrido determina la lista de particulas a leer de cada DF
        listado_ind     = [] 
        particle_names  = []
    
        for n in tqdm(range(len(files_DFs)), desc='Part. list', leave = False, disable = ~np.array(print_progress)):
                df = pd.read_pickle((files_DFs[n]))
                # Numero de frames no NaNs
                nframes = df.count().values
                cond = (nframes > length_threshold)
                listado_ind.append(df.columns[cond])
                particle_names.extend(df.columns[cond].values) 
    
        # create BIG dataframe with right columns (particle_names)
        dfout = pd.DataFrame(columns=particle_names, index=df.index, dtype=float)
    
        # fill the dataframe
        for n in tqdm(range(len(files_DFs)), desc='Filtered DF', leave = False, disable = ~np.array(print_progress)):
                if listado_ind[n].empty is not True:
                        df = pd.read_pickle((files_DFs[n]))
                        dfout.loc[:, listado_ind[n].values] = df.loc[:,listado_ind[n].values]
    
        return dfout, ID

#--------------------------------------------

if __name__ == "__main__":

    # Load in image paths
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('input_folder', type=str)
    args = argparser.parse_args()
    
    dirname="combined_"
    output_file = os.path.join(args.input_folder, dirname)
     
    Lmin = 1000
    DF, ID  = generar_dataframe_filtrado(args.input_folder, length_threshold = Lmin, print_progress = True)
    DF.to_pickle(output_file + ID + '.out') 
