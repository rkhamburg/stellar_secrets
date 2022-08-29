import numpy as np
from astropy.io import ascii
from plotting import get_bins

def get_data(filename, column='flux_1024', type='all', data_type='peakflux'):

    # Read data from file
    data = read_data(filename, column=column, type=type)

    # Get histogram counts from data
    data_counts = get_data_counts(data, data_type=data_type)
    data_counts[data_counts==0] = 1e-3

    return data, data_counts



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
