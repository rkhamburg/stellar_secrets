import numpy as np
from astropy.io import ascii
from plotting import get_bins

def get_data(filename, column='flux_1024', type='all', data_type='peakflux'):

    # Read data from file
    data  = read_data(filename, column=column, type=type)

    # Get histogram counts from data
    data_counts = get_data_counts(data, data_type=data_type)
    data_counts[data_counts==0] = 1e-3

    return data, data_counts


def get_luminosity_data(filename, t90_file=None, type='all'):
    # Read file with rest-frame parameters
    luminosity_file = ascii.read(filename)
    
    # Read GBM trigger file
    if type != 'all':
        gbm_triggers = read_data(t90_file, type=type, column='name')
        gbm_triggers = [int(grb[3:]) for grb in gbm_triggers]
    else:
        gbm_triggers = luminosity_file["col2"]

    # Grab luminosities
    luminosities = []
    for line in luminosity_file:
        if line[1] in gbm_triggers:
            if type == 'long' and line[1] != 170817529:
                luminosities.append(line["col13"])
            elif type == 'short' or type == 'all':
                luminosities.append(line["col13"])
        if type == 'short' and line[1] == 170817529:
            luminosities.append(line["col13"])

    # Histogram the luminosities
    #print (len(luminosities))
    return np.histogram(luminosities, bins=get_bins('luminosity'))[0]


def read_data(filename, type='all', column='flux_1024'):

    data = ascii.read(filename, header_start=0, delimiter='|')

    if type == 'long':
        data = data[np.where(data["t90"]>2.)]
    elif type == 'short':
        data = data[np.where(data["t90"]<2.)]

    return data[column]



def get_data_counts(data, data_type='peakflux'):

    # Define bins to be used in the histogram
    bins = get_bins(data_type)
    # Scale data to length of observations per year
    # July 14 2008 - June 30 2022: ~14 years
    weights = [1./14.] * len(data)
    # Make a histogram of data to get bin counts
    data_counts = np.histogram(data, bins=bins, weights=weights)[0]

    return data_counts
