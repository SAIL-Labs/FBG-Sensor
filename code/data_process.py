import pandas as pd

def psg_to_numpy(filepath, verbose=True):

    """
    This function takes a psg_trn.txt file generated from the NASA-GSF Planetary Spectru Gemerator (PSG, Villanueva et al. 2018, 2022, https://psg.gsfc.nasa.gov/) and converts the data into a numpy array. The first column is the wave/freq, the second column is the total and the following columns correspond to each gas species.
    
    :Parameters:
        **filepath: string**

        Path to the psg file.

    :Optional parameters:

        **verbose: boolean**

        If True, displays the metadata information in the file.

    """
    with open(filepath, "r") as f:
        file = f.read().strip()
        file = file.split('\n')

    skip = 0
    for line in file:
        if line[0] == '#':
            if verbose:
                print(line)
            skip += 1

    data = pd.read_csv(filepath, sep=' ', skiprows=skip, header=None).to_numpy()

    return data


