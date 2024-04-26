import numpy as np
import pandas as pd

def lists_are_equal(list1, list2):
    if (len(list1) != len(list2)):
        return False
    elif (list1.sort() != list2.sort()):
        return False
    else:
        return True


def get_trajectories(df_input, only_del_nans = False):
    """
    Input: pandas df_file computed by the ptv code
    Output: xP, yP arrays containing the positions in time row-wise and the different particles column wise.
    """
    
    df = pd.read_pickle(df_input)
    
    Ntray = int(df.shape[1]//2) 
    Ntime = len(df.iloc[:, 0])
    
    xP = np.zeros((Ntray, Ntime))
    yP = np.zeros((Ntray, Ntime))
    
    # Convert pd datastructure into np array
    for n in range(Ntray):
        x = df.iloc[:, 2*n  ].values.astype(float)
        y = df.iloc[:, 2*n+1].values.astype(float)
        xP[n] = x
        yP[n] = y

    # Divide data into tranches of non-changing particle count in the trajectory data
    bins = 0
    idx = []
    nan_indices = [] 
    nan_idx_old = []
    for i, row in enumerate(xP.T):
        nan_idx_new = np.where(np.isnan(row) == True)[0]
        if ((bins == 0) | (not lists_are_equal(nan_idx_old, nan_idx_new))):
            bins += 1
            idx.append(i)
            nan_indices.append(nan_idx_old)
            nan_idx_old = nan_idx_new


    # Get the largest batch of trajectories which are non-changing
    loc = np.argmax(np.diff(idx))
    max = idx[loc + 1] 
    min = idx[loc]
    nans = nan_indices[loc + 1]

    xP = xP[:, min:max] 
    yP = yP[:, min:max]  

    xP = np.delete(xP, (nans), axis=0)
    yP = np.delete(yP, (nans), axis=0)

    return xP, yP     


if __name__ == "__main__":
    import argparse

    # Load in image paths
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('input_file_df', type=str)
    args = argparser.parse_args()

    xP, yP = get_trajectories(args.input_file_df)

    print(xP.shape)
    print(yP.shape)

    import matplotlib.pyplot as plt
    plt.figure()
    for x, y in zip(xP, yP):
        plt.plot(x, y, '.-')
    plt.show()
    

