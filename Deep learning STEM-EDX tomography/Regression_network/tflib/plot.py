import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import _pickle as pickle
import os

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_epoch = [-1]
def tick():
        _epoch[0] += 1

def plot(name, value):
        _since_last_flush[name][_epoch[0]] = value

def flush(experiment=''):
    prints = []

    for name in sorted(_since_last_flush.keys()):
        vals = _since_last_flush[name]

        prints.append("{}\t{:.4f}".format(name, np.mean(np.array(list(vals.values())))))
        _since_beginning[name].update(vals)

        x_vals = np.sort(np.array(list(_since_beginning[name].keys())))
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.savefig(os.path.join(experiment, name.replace(' ', '_')+'.jpg'))

    print("Epoch {}\t{}".format(_epoch[0], "\t".join(prints)))
    _since_last_flush.clear()

