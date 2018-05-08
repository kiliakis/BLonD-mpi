#!/usr/bin/python
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import fnmatch
from plot.plotting_utilities import *

import argparse


parser = argparse.ArgumentParser(description='Report the avg time spend on communication and computation.',
                                 usage='python script.py [-p file_pattern] [-i indir] [-o outfile]')

parser.add_argument('-p', '--pattern', type=str, default='report-worker-*.csv',
                    help='The report file names pattern. '
                    ' Default: report-worker-*.csv')

parser.add_argument('-o', '--outfile', type=str, default=sys.stdout,
                    help='The file to save the report.'
                    ' Default: Print to the stdout')

parser.add_argument('-i', '--indir', type=str, default='./',
                    help='The directory containing the report files.'
                    ' Default: Use the current working directory.')



# def plot_pie(tc, input_file):
#     print(input_file)
#     data = np.genfromtxt(input_file, dtype=str, delimiter='\t')
#     header = data[0]
#     data = data[1:]
#     keys = data[:, 0].tolist()
#     values = data[:, 1].tolist()
#     CPI = values[keys.index('CPI')]
#     del values[keys.index('CPI')]
#     keys.remove('CPI')
#     values = np.array(values, float)

#     # plt.figure(figsize=(6.5, 4))
#     plt.figure()
#     # plt.grid(True, which='major', alpha=0.5)
#     plt.xlabel(title.format(tc, CPI), fontsize=11)
#     cmap = plt.get_cmap('jet')
#     colors = cmap(np.linspace(0., 1., len(keys)))
#     explode = [0] * len(keys)
#     patches, texts, autotexts = plt.pie(values, shadow=False, colors=colors,
#                                         counterclock=False,
#                                         autopct='%1.1f%%',
#                                         textprops={'fontsize': '10'},
#                                         startangle=0,
#                                         explode=explode)
#     for t in autotexts:
#         if(float(t.get_text().split('%')[0]) < 3):
#             t.set_text('')
#     # autotexts[0].set_color('w')
#     plt.axis('equal')
#     # plt.subplot(grid[0, 0])
#     plt.legend(keys, loc='upper center', fancybox=True,
#                framealpha=0.4, ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.05))
#     # plt.legend(labels, loc='upper center', bbox_to_anchor=(-0.6, 2.4), ncol=5,
#     #            fancybox=True, fontsize=8, framealpha=0.5)
#     plt.tight_layout()
#     if show:
#         plt.show()
#     else:
#         img = image_name.format(
#             tc, input_file.split('/')[-1].split('.csv')[0])
#         plt.savefig(img, bbox_inches='tight')

#     plt.close()


if __name__ == '__main__':
    args = parser.parse_args()
    file_pattern = args.pattern
    indir = args.indir
    files = fnmatch.filter(os.listdir(indir), file_pattern)
    comm_l = []
    comp_l = []
    other_l = []
    for f in files:
        # print(f)
        data = np.genfromtxt(f, dtype=str, delimiter='\t')
        header = data[0]
        total_time = data[-1]
        data = data[1:-1]
        # print(data)
        comm = np.sum([float(r[-1]) for r in data if 'comm' in r[0]])
        comp = np.sum([float(r[-1]) for r in data if 'comp' in r[0]])
        other = np.sum([float(r[-1]) for r in data if 'Other' in r[0]])
        # print('Comm:', comm)
        # print('Comp:', comp)
        comm_l.append(comm)
        comp_l.append(comp)
        other_l.append(other)
    string = 'type\tavg\tmin\tmax\tstd\n'
    string += ('%s\t%.2f\t%.2f\t%.2f\t%.2f\n' % 
        ('comm', np.mean(comm_l), np.min(comm_l), np.max(comm_l), np.std(comm_l))) 
    string += ('%s\t%.2f\t%.2f\t%.2f\t%.2f\n' % 
        ('comp', np.mean(comp_l), np.min(comp_l), np.max(comp_l), np.std(comp_l))) 
    string += ('%s\t%.2f\t%.2f\t%.2f\t%.2f\n' % 
        ('other', np.mean(other_l), np.min(other_l), np.max(other_l), np.std(other_l))) 

    if args.outfile == sys.stdout:
        sys.stdout.write(string)
    else:
        open(args.outfile, 'w').write(string)
