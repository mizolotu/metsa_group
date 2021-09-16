import numpy as np

from matplotlib import pyplot as pp

def plot_line(x, y, marker, xlabel, ylabel, fpath):
    pp.figure(figsize=(16, 12))
    pp.plot(x, y, marker)
    pp.xlabel(xlabel, fontdict={'size': 12})
    pp.ylabel(ylabel, fontdict={'size': 12})
    pp.xticks(fontsize=12)
    pp.yticks(fontsize=12)
    pp.savefig(fpath, bbox_inches='tight')
    pp.close()

def plot_bars(items, heights, hatches, items_for_argsort, figh, xlabel, ylabel, legend_items, legend_names, fpath, sort=False, reverse=False, xticks_rotation='horizontal', plot_png=False):
    if sort:
        idx = np.argsort(items_for_argsort)
        if reverse:
            idx = idx[::-1]
        items = np.array(items)[idx].tolist()
        heights = np.array(heights)[idx].tolist()
        hatches = np.array(hatches)[idx].tolist()
    pp.figure(figsize=(21.2, figh))
    pp.bar(items, height=heights, color='white', edgecolor='black', hatch=hatches)
    pp.xlabel(xlabel, fontdict={'size': 12})
    pp.ylabel(ylabel, fontdict={'size': 12})
    pp.xticks(fontsize=12, rotation=xticks_rotation)
    pp.yticks(fontsize=12)
    if len(legend_items) > 0 and len(legend_names) > 0:
        pp.legend(legend_items, legend_names, prop={'size': 12})
    pp.savefig(fpath, bbox_inches='tight')
    if plot_png:
        fpath = fpath.replace('.pdf', '.png')
        pp.savefig(fpath, bbox_inches='tight')
    pp.close()