#!/usr/bin/python
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import fnmatch
import csv

# from plot.plotting_utilities import *

import argparse


parser = argparse.ArgumentParser(description='Report the avg time spend on communication and computation.',
                                 usage='python script.py [-p file_pattern] [-i indir] [-o outfile]')

parser.add_argument('-p', '--pattern', type=str, default='report-worker-*.csv',
                    help='The report file names pattern. '
                    ' Default: report-worker-*.csv')

parser.add_argument('-o', '--outfile', type=argparse.FileType('w'),
                    default=sys.stdout,
                    help='The file(s) to save the report.'
                    ' Default: Print to the stdout')

parser.add_argument('-i', '--indir', type=str, default='./',
                    help='The directory containing the report files.'
                    ' Default: Use the current working directory.')


parser.add_argument('-r', '--report', type=str, choices=['comm-comp', 'avg'],
                    default='comm-comp',
                    help='Choose from 2 report types: comm-comp or avg.'
                    ' Default: comm-comp.')


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

def report_comm_comp(indir, files, outfile):
    comm_l = []
    comp_l = []
    other_l = []
    total_l = []
    serial_l = []
    overhead_l = []
    for f in files:
        # print(f)
        data = np.genfromtxt(indir+'/'+f, dtype=str, delimiter='\t')
        header = data[0]
        # total_time = data[-1]
        data = data[1:]
        # print(data)
        comm = np.sum([(float(r[-1]), float(r[1]))
                       for r in data if 'comm' in r[0]], axis=0)
        comp = np.sum([(float(r[-1]), float(r[1]))
                       for r in data if 'comp' in r[0]], axis=0)
        overhead = np.sum([(float(r[-1]), float(r[1]))
                           for r in data if 'overhead' in r[0]], axis=0)
        serial = np.sum([(float(r[-1]), float(r[1]))
                         for r in data if 'serial' in r[0]], axis=0)
        other = np.sum([(float(r[-1]), float(r[1]))
                        for r in data if 'Other' in r[0]], axis=0)
        total = np.sum([(float(r[-1]), float(r[1]))
                        for r in data if 'total_time' in r[0]], axis=0)
        # print(total)
        # print('Comm:', comm)
        # print('Comp:', comp)
        comm_l.append(comm)
        comp_l.append(comp)
        other_l.append(other)
        serial_l.append(serial)
        overhead_l.append(overhead)
        total_l.append(total)
    string = 'type\tavg_time(sec)\tavg_percent\tmin\tmax\tstd\n'
    try:
        string += ('%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' %
                   ('comm', np.mean(comm_l, axis=0)[1], np.mean(comm_l, axis=0)[0],
                    np.min(comm_l, axis=0)[0], np.max(comm_l, axis=0)[0], np.std(comm_l, axis=0)[0]))
    except:
        pass

    try:
        string += ('%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' %
                   ('comp', np.mean(comp_l, axis=0)[1], np.mean(comp_l, axis=0)[0],
                    np.min(comp_l, axis=0)[0], np.max(comp_l, axis=0)[0], np.std(comp_l, axis=0)[0]))
    except:
        pass

    try:
        string += ('%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' %
                   ('serial', np.mean(serial_l, axis=0)[1], np.mean(serial_l, axis=0)[0],
                    np.min(serial_l, axis=0)[0], np.max(serial_l, axis=0)[0], np.std(serial_l, axis=0)[0]))
    except:
        pass

    try:
        string += ('%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' %
                   ('overhead', np.mean(overhead_l, axis=0)[1], np.mean(overhead_l, axis=0)[0],
                    np.min(overhead_l, axis=0)[0], np.max(overhead_l, axis=0)[0], np.std(overhead_l, axis=0)[0]))
    except:
        pass

    try:
        string += ('%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' %
                   ('other', np.mean(other_l, axis=0)[1], np.mean(other_l, axis=0)[0],
                    np.min(other_l, axis=0)[0], np.max(other_l, axis=0)[0], np.std(other_l, axis=0)[0]))
    except:
        pass
    
    try:
        string += ('%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' %
                   ('total', np.mean(total_l, axis=0)[1], np.mean(total_l, axis=0)[0],
                    np.min(total_l, axis=0)[0], np.max(total_l, axis=0)[0], np.std(total_l, axis=0)[0]))
    except:
        pass

    outfile.write(string)


def report_avg(indir, files, outfile):
    default_funcs = []
    default_header = []
    acc_data = []
    num = 0
    for f in files:
        # print(f)
        data = np.genfromtxt(indir+'/'+f, dtype=str, delimiter='\t')
        header = data[0]
        data = data[1:]
        funcs = data[:, 0]
        data = data[:, 1:]
        data = np.array(data, float)
        if len(default_funcs) == 0:
            default_funcs = funcs
        elif not np.array_equal(default_funcs, funcs):
            print('Problem with file: ', indir+'/'+f)
            continue

        if len(default_header) == 0:
            default_header = header
        elif not np.array_equal(default_header, header):
            print('Problem with file: ', indir+'/'+f)
            continue

        if len(acc_data) == 0:
            acc_data = data
        else:
            acc_data += data
        num += 1
    try:
        acc_data = np.around(acc_data/num, 2)
    except TypeError as e:
        print('[Error] with dir ', indir)
        return

    writer = csv.writer(outfile, delimiter='\t')
    writer.writerow(default_header)
    for f, r in zip(default_funcs, acc_data):
        writer.writerow([f]+list(r))


if __name__ == '__main__':
    args = parser.parse_args()
    # print(args)
    file_pattern = args.pattern
    indir = args.indir
    files = fnmatch.filter(os.listdir(indir), file_pattern)
    outfile = args.outfile
    if outfile == 'sys.stdout':
        outfile = sys.stdout

    if args.report == 'comm-comp':
        report_comm_comp(indir, files, outfile)
    elif args.report == 'avg':
        report_avg(indir, files, outfile)
