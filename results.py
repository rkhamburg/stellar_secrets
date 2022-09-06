import bilby
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import argparse

from priors import *
from plotting import corner_plots

def find_labels(filename):
    parsed_filename = filename.split('/')
    info = parsed_filename[-1].split('_')
    prior_info = make_prior_joint(info[0], 'SFR', info[1], float(info[2]))
    prior_info = prior_info.items()
    labels = [prior[1].latex_label for prior in prior_info]
    labels = [l for l in labels if l != None]
    return labels

class Results():
    def __init__(self, output_file, truths=None, name=False, save=False):
        self.file = output_file
        self.truths = truths
        self.name = name
        self.save = save
        return

    def read_output(self):
        """ Read various bilby/dynesty output files """
        if self.file[-5:] == '.json':
            self.read_json()
        elif self.file[-4:] == '.dat':
            self.read_dat()
        #elif self.file[-7:] == '.pickle':
        #    self.read_pickle()
        self.transform_samples = self.samples.T
        return self.samples

    def read_json(self):
        """ Read bilby/dynesty json output file """
        samples = bilby.result.read_in_result(self.file)
        self.labels = samples.parameter_labels
        samples = samples.posterior
        samples = np.array([samples[s].values for s in samples])
        self.samples = samples[:-2].T
        self.truths = samples.injection_parameters
        return

    def read_dat(self):
        """ Read bilby/dynesty dat output file """
        with open(self.file) as f:
            lines = f.readlines()
        #self.header = lines[0]
        samples = []
        for i in range(1, len(lines)):
            line = lines[i].split()
            row = []
            for j in range(len(line)):
                row.append(float(line[j]))
            samples.append(row)
        self.samples = np.array(samples)
        return self.samples

    def read_pickle(self):
        """ Read bilby/dynesty pickle output file """
        with open(self.file, 'rb') as f:
            results = pickle.load(f)
        #results_dict = results.__dict__
        #for r in results_dict:
        #    print (r)
        return #self.samples

    def get_map_values(self):
        self.maps = self.samples[-1]
        return self.maps

    def get_median_values(self):
        self.medians = np.median(self.transform_samples, axis=1)
        return self.medians

    def get_labels(self):
        return self.labels


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', default=None, help='path to output file')
parser.add_argument('-s', '--save', type=bool, default=False, help='save plots')
parser.add_argument('-n', '--name', default='output', help='Name for plots')
parser.add_argument('-t','--truths', nargs='+', default=None, help='Truths array')
parser.add_argument('-c','--corner', type=bool, default=False,
    help='Make posterior corner plots')
parser.add_argument('-p','--predict', type=bool, default=False,
    help='Make posterior predictive plots')
args = parser.parse_args()


# Change truth dict into floats
if args.truths is not None:
    truths = [float(t) for t in args.truths]
else:
    truths = None

r = Results(args.file, truths, args.name, args.save)
posterior_samples = r.read_output()
maps = r.get_map_values()
medians = r.get_median_values()

# Get labels
if args.file[-5:] == '.json':
    labels = r.get_labels()
else:
    labels = find_labels(args.file)

# Make corner plot
if args.corner is not False:
    print ('making corner plots...')
    corner_plots(posterior_samples, bins=10, map=maps, medians=medians,
        truths=truths, labels=labels, save=args.save, name=args.name,
        smoothing=1.5, hcolor='#018571', truth_color='tomato')

if args.predict is not False:
    print ('making posterior predictive plots...')
