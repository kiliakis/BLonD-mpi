#!/usr/bin/python
import os
import numpy as np
import sys
import fnmatch
# import csv
import matplotlib.pyplot as plt
from plot.plotting_utilities import *
import argparse
import pickle

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
this_filename = sys.argv[0].split('/')[-1]


parser = argparse.ArgumentParser(description='Generate worker trace from pickle files.',
                                 usage='python script.py [-i indir] [-p pattern]')

parser.add_argument('-p', '--pattern', type=str, default='worker-*.log',
                    help='The input file names pattern. '
                    ' Default: worker-*.log')

parser.add_argument('-o', '--outdir', type=str,
                    default=None,
                    help='The directory to store the traces.'
                    ' Default: Same as the input directory')

parser.add_argument('-f', '--filename', type=str,
                    default=None,
                    help='The output filename.'
                    ' Default: Automatically assigned.')


parser.add_argument('-i', '--indir', type=str, default='./',
                    help='The directory containing the report files.'
                    ' Default: Use the current working directory.')

parser.add_argument('-skip', '--skip', type=int, default=5,
                    help='How many points to skip'
                    ' Default: 5, plot every 5 points.')



# parser.add_argument('-w', '--window', type=int, default=10,
#                     help='Width of running mean to smoothen spiky curves.'
#                     ' Default: 10.')


# parser.add_argument('-from', '--from', type=str, default='log', choices=['pickle', 'log'],
#                     dest='fromfile',
#                     help='Use log files as input or pickle files.'
#                     ' Default: log.')


# parser.add_argument('-r', '--report', type=str, choices=['comm-comp', 'avg', 'delta'],
#                     default='comm-comp',
#                     help='Choose from 3 report types: comm-comp, avg, delta'
#                     ' Default: comm-comp.')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots.')
gconfig = {
    'lines': {
        'comp': r'comp:(\w+)',
        'comm': r'comm:(\w+)',
        'const': r'^(?!\w+:(sync|intraSync))serial:(\w+)',
        'sync': r'serial:(sync|intraSync)'
    },
    'colors': {
        'comp': 'blue',
        'comm': 'orange',
        'const': 'red',
        'sync': 'gray',
        # 'total': 'black'
    },
    'alpha': {
        'total': 0.7
    },
    'extract_turns': 'comp:histo',
    'plot': {
        # 'lw': 1,
        'ls': '',
        'marker': '.'

    },
    'xlabel': {
        'xlabel': 'Turns',
        'labelpad': 1,
        'fontsize': 10
    },
    'ylabel': {
        'ylabel': 'Turn time (ms)',
        'labelpad': 1,
        'fontsize': 10
    },
    'title': {
        'fontsize': 10,
        # 'y': 0.95,
        'fontweight': 'bold',
    },
    'annotate': {
        'fontsize': 9,
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center'
    },
    'xticks': {'fontsize': 10, 'rotation': '0', 'fontweight': 'bold'},
    'ticks': {'fontsize': 10, 'rotation': '0'},
    'legend': {
        'loc': 'upper right', 'ncol': 1, 'handlelength': 1.1, 'fancybox': True,
        'framealpha': 0., 'fontsize': 9, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.2, 'borderaxespad': 0.1, 'columnspacing': 0.3,
        'bbox_to_anchor': (1.35, 1)
    },
    'subplots_adjust': {
        'wspace': 0.05, 'hspace': 0.1,
        # 'top': 1
    },
    'tick_params': {
        'pad': 1, 'top': 0, 'bottom': 0, 'left': 1,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'fontname': 'DejaVu Sans Mono',
    # 'ylim': [0.45, 1.02],
    # 'yticks': [0.5, 0.6, 0.7, .8, .9, 1.],
    'outfiles': ['{}/{}-corr.png', '{}/{}-corr.pdf'],
    'subplots': {'sharex': True, 'sharey': True, 'ncols': 2,
                 'figsize': [5, 2.5]}
}

plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
# Force sans-serif math mode (for axes labels)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath']
plt.rcParams['font.family'] = 'sans-serif'  # ... for regular text
plt.rcParams['font.sans-serif'] = 'Helvetica'




def plot_correlations_from_log(ax, file, idx, nrows):
    plt.sca(ax)

    regexp = re.compile(
        r'.*\[(\d+)\]:\sTurn\s(\d+),\sTconst\s([^,]+),\sTcomp\s([^,]+),\sTcomm\s([^,]+),\sTsync\s([^,]+),\sLatency\s([^,]+),\sParticles\s([^,]+).*')
    with open(file, 'r') as f:
        plotdir = {}
        for line in f:
            match = regexp.match(line)
            if match:
                wid, turn, const, comp, comm, sync, latency, particles = match.groups()
                if 'turns' not in plotdir:
                    plotdir['turns'] = []
                    plotdir['wid'] = wid
                    plotdir['const'] = []
                    plotdir['comm'] = []
                    plotdir['comp'] = []
                    plotdir['sync'] = []
                    plotdir['latency'] = []
                    plotdir['particles'] = []
                    plotdir['total'] = []
                # plotdir['wid'].append(wid)
                assert plotdir['wid'] == wid
                plotdir['turns'].append(int(turn))
                plotdir['const'].append(float(const))
                plotdir['comp'].append(float(comp))
                plotdir['comm'].append(float(comm))
                plotdir['sync'].append(float(sync))
                plotdir['latency'].append(float(latency))
                plotdir['particles'].append(float(particles))
                plotdir['total'].append(
                    float(const) + float(comp) + float(sync) + float(comm))

    for k, v in plotdir.items():
        if k not in gconfig['colors']:
            continue
        # x = np.arange(len(v))[::args.skip]
        # y = v[::args.skip]
        # y = running_mean(v, args.window)
        v = np.array(v)
        v = (v - min(v)) / ((max(v) - min(v)))
        print(f'{k}: {v}')
        plt.plot(plotdir['particles'], v, color=gconfig['colors'][k],
                 label=k, **gconfig['plot'], alpha=gconfig['alpha'].get(k, 1))

    # plt.ylim(ymax=1.4 * np.mean(plotdir['total']))
    # plt.xlim(0-1.3*width/2, pos-1.4*width/2)
    plt.grid(True, which='both', axis='y', alpha=0.5)

    plt.yticks(**gconfig['ticks'])
    plt.xticks(**gconfig['xticks'])

    if idx // gconfig['subplots']['ncols'] == nrows - 1:
        plt.xlabel(**gconfig['xlabel'])

    if idx % gconfig['subplots']['ncols'] == 0:
        plt.ylabel(**gconfig['ylabel'])

    if idx % gconfig['subplots']['ncols'] == gconfig['subplots']['ncols'] - 1:
        plt.legend(**gconfig['legend'])
    # plt.title(os.path.splitext(file)[0].split('/')[-1],
        # **gconfig['title'])
    worker = os.path.splitext(file)[0].split('/')[-1]
    # plt.text(0.98, .98, '{} total: {:.2f}s'.format(worker, np.sum(plotdir['total'])/1e3),
    #          ha='right', va='top',
    #          transform=ax.transAxes)
    ax.tick_params(**gconfig['tick_params'])
    plt.tight_layout()


if __name__ == '__main__':
    args = parser.parse_args()
    file_pattern = args.pattern
    indir = args.indir
    files = fnmatch.filter(os.listdir(indir), file_pattern)
    outdir = args.outdir
    if not outdir:
        outdir = indir
    os.makedirs(outdir, exist_ok=True)
    nrows = int(np.ceil(len(files)/gconfig['subplots']['ncols']))
    fig, axes = plt.subplots(nrows=nrows, **gconfig['subplots'])
    axes = axes.ravel()
    for file, ax, idx in zip(sorted(files), axes, np.arange(len(files))):
        plot_correlations_from_log(ax, os.path.join(indir, file), idx, nrows)

    # plt.tight_layout()
    plt.subplots_adjust(**gconfig['subplots_adjust'])
    for file in gconfig['outfiles']:
        if args.filename:
            file = file.format(outdir, args.filename)
        else:
            file = file.format(
                outdir, os.path.basename(os.path.normpath(indir)))
        # print('[{}] {}: {}'.format(this_filename[:-3], 'Saving figure', file))
        save_and_crop(fig, file, dpi=300, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()
