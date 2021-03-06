import numpy as np
import pandas as pd

from matplotlib import pyplot as pp

def plot_multiple_lines(xs, ys, markers, xlabel, ylabel, fpath):
    pp.figure(figsize=(16, 12))
    for x, y, marker in zip(xs, ys, markers):
        pp.plot(x, y, marker)
    pp.xlabel(xlabel, fontdict={'size': 12})
    pp.ylabel(ylabel, fontdict={'size': 12})
    pp.xticks(fontsize=12)
    pp.yticks(fontsize=12)
    pp.savefig(fpath, bbox_inches='tight')
    pp.close()

def plot_line(x, y, marker, xlabel, ylabel, fpath):
    pp.figure(figsize=(16, 12))
    pp.plot(x, y, marker)
    pp.xlabel(xlabel, fontdict={'size': 12})
    pp.ylabel(ylabel, fontdict={'size': 12})
    pp.xticks(fontsize=12)
    pp.yticks(fontsize=12)
    pp.savefig(fpath, bbox_inches='tight')
    pp.close()

def plot_multiple_bars(items, heights, hatches, figh, xlabel, ylabel, legend_items, legend_names, fpath, xticks_rotation='horizontal', plot_png=True, width=0.1):
    pp.figure(figsize=(21.2, figh))
    x = 1 + np.arange(len(items))
    steps = np.arange(len(items)) * width
    for s, he, ha in zip(steps, heights, hatches):
        pp.bar(x + s, height=he, width=width, color='white', edgecolor='black', hatch=ha)
    pp.xlabel(xlabel, fontdict={'size': 12})
    pp.ylabel(ylabel, fontdict={'size': 12})
    pp.xticks(x + width * (len(items) - 1) / 2, items, fontsize=12, rotation=xticks_rotation)
    pp.yticks(fontsize=12)
    if len(legend_items) > 0 and len(legend_names) > 0:
        pp.legend(legend_items, legend_names, prop={'size': 12})
    pp.savefig(fpath, bbox_inches='tight')
    if plot_png:
        fpath = fpath.replace('.pdf', '.png')
        pp.savefig(fpath, bbox_inches='tight')
    pp.close()

def plot_bars(items, heights, hatches, items_for_argsort, figh, xlabel, ylabel, legend_items, legend_names, fpath, sort=False, reverse=True, xticks_rotation='horizontal', plot_png=True, figw=21.2, nan_mask=0):
    non_nan_idx = np.where(~pd.isna(heights))
    heights[np.where(pd.isna(heights))] = nan_mask
    ydelta = np.max(heights[non_nan_idx]) - np.min(heights[non_nan_idx])
    hmin, hmax = np.min(heights[non_nan_idx]) - 0.05 * ydelta, np.max(heights[non_nan_idx]) + 0.05 * ydelta
    items_for_argsort[np.where(pd.isna(items_for_argsort))] = nan_mask
    if sort:
        idx = np.argsort(items_for_argsort)
        if reverse:
            idx = idx[::-1]
        items = np.array(items)[idx].tolist()
        heights = np.array(heights)[idx].tolist()
        hatches = np.array(hatches)[idx].tolist()
    print(f'First item and value at {fpath}: {items[0]} = {heights[0]}')
    pp.figure(figsize=(figw, figh))
    pp.bar(items, height=heights, color='white', edgecolor='black', hatch=hatches)
    pp.xlabel(xlabel, fontdict={'size': 12})
    pp.ylabel(ylabel, fontdict={'size': 12})
    pp.xticks(fontsize=12, rotation=xticks_rotation)
    pp.yticks(fontsize=12)
    pp.ylim([hmin, hmax])
    if len(legend_items) > 0 and len(legend_names) > 0:
        pp.legend(legend_items, legend_names, prop={'size': 12})
    pp.savefig(fpath, bbox_inches='tight')
    if plot_png:
        fpath = fpath.replace('.pdf', '.png')
        pp.savefig(fpath, bbox_inches='tight')
    pp.close()