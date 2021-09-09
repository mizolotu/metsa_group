from matplotlib import pyplot as pp
from itertools import cycle

def plot_markers(x, y, marker, xlabel, ylabel, fpath):
    pp.figure(figsize=(16, 12))
    pp.plot(x, y, marker)
    pp.xlabel(xlabel, fontdict={'size': 12})
    pp.ylabel(ylabel, fontdict={'size': 12})
    pp.xticks(fontsize=12)
    pp.yticks(fontsize=12)
    pp.savefig(fpath, bbox_inches='tight')
    pp.close()
