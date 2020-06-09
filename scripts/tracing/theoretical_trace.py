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

# parser.add_argument('-p', '--pattern', type=str, default='worker-*.p',
#                     help='The pickle file names pattern. '
#                     ' Default: worker-*.p')

parser.add_argument('-o', '--outdir', type=str,
                    default=None,
                    help='The directory to store the traces.'
                    ' Default: Same as the input directory')

# parser.add_argument('-f', '--filename', type=str,
#                     default=None,
#                     help='The output filename.'
#                     ' Default: Automatically assigned.')


# parser.add_argument('-i', '--indir', type=str, default='./',
#                     help='The directory containing the report files.'
#                     ' Default: Use the current working directory.')

# parser.add_argument('-skip', '--skip', type=int, default=5,
#                     help='How many points to skip'
#                     ' Default: 5, plot every 5 points.')

# parser.add_argument('-w', '--window', type=int, default=10,
#                     help='Width of running mean to smoothen spiky curves.'
#                     ' Default: 10.')


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
        'total': 'black'
    },
    'alpha': {
        'total': 0.7
    },
    'extract_turns': 'comp:histo',
    'plot': {
        'lw': 1,
        'ls': '-',

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
        # 'bbox_to_anchor': (1.35, 1)
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
    'outfiles': ['{}/theoretical-traces.png', '{}/theoretical-traces.pdf'],
    'subplots': {'sharex': True, 'sharey': True, 'ncols': 3, 'nrows': 1,
                 'figsize': [7.5, 2.5]}
}

plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
# Force sans-serif math mode (for axes labels)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath']
plt.rcParams['font.family'] = 'sans-serif'  # ... for regular text
plt.rcParams['font.sans-serif'] = 'Helvetica'


def plot_traces_from_pickle(ax, file, idx, nrows):
    plt.sca(ax)

    with open(file, 'rb') as f:
        indir = pickle.load(f)
    plotdir = {}
    turns = len(indir[gconfig['extract_turns']])
    for k, v in indir.items():
        for line, expr in gconfig['lines'].items():
            match = re.match(expr, k)
            if match:
                if line not in plotdir:
                    plotdir[line] = np.zeros(turns)
                if len(v) != turns:
                    v = np.ones(turns) * np.sum(v) / turns
                plotdir[line] += v
                break
    plotdir['total'] = np.sum([v for v in plotdir.values()], axis=0)

    for k, v in plotdir.items():
        # x = np.arange(len(v))[::args.skip]
        # y = v[::args.skip]
        y = running_mean(v, args.window)
        plt.plot(np.arange(len(y)), y, color=gconfig['colors'][k],
                 label=k, **gconfig['plot'], alpha=gconfig['alpha'].get(k, 1))

    plt.ylim(ymax=1.4 * np.mean(plotdir['total']))
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
    plt.text(0.98, .98, '{} total: {:.2f}s'.format(worker, indir['total_time']/1e3),
             ha='right', va='top',
             transform=ax.transAxes)
    ax.tick_params(**gconfig['tick_params'])
    plt.tight_layout()


def plot_traces_from_log(ax, file, idx, nrows):
    plt.sca(ax)

    regexp = re.compile(
        '.*[(\d+)]:\sTurn\s(\d+),\sTconst\s(.*),\sTcomp\s(.*),\sTcomm\s(.*),\sTsync\s(.*),\sLatency\s(.*),\sParticles\s(.*)')
    f = open(file, 'r')

    plotdir = {}
    for line in file:
        match = regexp.match(line)
        if match:
            wid, turn, const, comp, comm, sync, latency, particles = match.groups()
            if 'turn' not in plotdir:
                plotdir['turn'] = []
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
            plotdir['turn'].append(int(turn))
            plotdir['const'].append(float(const))
            plotdir['comp'].append(float(comp))
            plotdir['comm'].append(float(comm))
            plotdir['sync'].append(float(sync))
            plotdir['latency'].append(float(latency))
            plotdir['particles'].append(int(particles))
            plotdir['total'].append(
                float(const) + float(comp) + float(sync) + float(comm))

    for k, v in plotdir.items():
        if k not in gconfig['colors']:
            continue
        # x = np.arange(len(v))[::args.skip]
        # y = v[::args.skip]
        # y = running_mean(v, args.window)
        plt.plot(plotdir['turns'], v, color=gconfig['colors'][k],
                 label=k, **gconfig['plot'], alpha=gconfig['alpha'].get(k, 1))

    plt.ylim(ymax=1.4 * np.mean(plotdir['total']))
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
    plt.text(0.98, .98, '{} total: {:.2f}s'.format(worker, indir['total_time']/1e3),
             ha='right', va='top',
             transform=ax.transAxes)
    ax.tick_params(**gconfig['tick_params'])
    plt.tight_layout()


if __name__ == '__main__':
    args = parser.parse_args()
    outdir = args.outdir
    if not outdir:
        outdir = indir
    os.makedirs(outdir, exist_ok=True)

    orig = 100

    init = 500
    incr = 200
    top = 1000
    dcr = 200
    delayed = 150

    delay1_theory = np.array([orig]*init + list(np.linspace(orig, delayed, incr)) + [
                             delayed] * top + list(np.linspace(delayed, orig, dcr)) + [orig]*(init))
    sync1_theory = np.array([orig/10] * len(delay1_theory))
    total1_theory = delay1_theory + sync1_theory

    delay2_theory = [orig]*len(delay1_theory)

    sync2_theory = total1_theory - delay2_theory
    # p.array([orig/10]*init + list(np.arange(orig/10, delayed-orig+1, 1)) + [delayed-orig] * top + list(np.arange(delayed-orig, orig/10 - 1, -1)) + [orig/10]*(init//2))
    # sync2_theory = np.array([10]*1000 + list(np.arange(10, 61, 1)) + [61] * 2000 + list(np.arange(60, 9, -1)) + [10]*100)
    total2_theory = delay2_theory + sync2_theory

    delay_optimal = (delay1_theory + delay2_theory) / 2
    sync_optimal = np.copy(sync1_theory)
    total_optimal = delay_optimal + sync_optimal

    x = np.arange(len(delay1_theory))

    fig, ax_arr = plt.subplots(**gconfig['subplots'])

    ax = ax_arr[0]
    plt.sca(ax)
    plt.title('Worker 0')
    plt.ylabel('Turn time')
    plt.xlabel('Turns')

    plt.plot(x, delay1_theory, ls='-', color='blue', label='comp')
    plt.plot(x, sync1_theory, ls='-', color='red', label='sync')
    plt.plot(x, total1_theory, ls='-', color='black', label='total')
    # plt.plot(x, delay_optimal, ls='--', color='blue', label='optimal-delay')
    # plt.plot(x, sync_optimal, ls='--', color='blue', label='optimal-sync')
    plt.yticks([0, 25, 50, 75, 100, 125, 150]   )

    plt.scatter([init, init+incr, init+incr+top, init+incr+top+dcr],
                [orig, delayed, delayed, orig],
                s=30, edgecolor='black', facecolor='1')

    # ax.annotate('')

    plt.tight_layout()
    plt.legend(**gconfig['legend'])

    ax = ax_arr[1]
    plt.sca(ax)
    plt.title('Worker 1')
    # plt.ylabel('Delay(ms)')
    plt.xlabel('Turns')

    plt.plot(x, delay2_theory, ls='-', color='blue', label='comp')
    plt.plot(x, sync2_theory, ls='-', color='red', label='sync')
    plt.plot(x, total2_theory, ls='-', color='black', label='total')

    plt.tight_layout()
    plt.legend(**gconfig['legend'])

    ax = ax_arr[2]
    plt.sca(ax)
    plt.title('Balanced (all workers)')
    # plt.ylabel('Delay(ms)')
    plt.xlabel('Turns')

    plt.plot(x, delay_optimal, ls='-', color='blue', label='comp')
    plt.plot(x, sync_optimal, ls='-', color='red', label='sync')
    plt.plot(x, total_optimal, ls='-', color='black', label='total')

    plt.tight_layout()
    plt.legend(**gconfig['legend'])

    plt.subplots_adjust(**gconfig['subplots_adjust'])
    for file in gconfig['outfiles']:
        file = file.format(outdir)
        # print('[{}] {}: {}'.format(this_filename[:-3], 'Saving figure', file))
        save_and_crop(fig, file, dpi=300, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()
