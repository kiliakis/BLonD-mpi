#!/usr/bin/python
import os
import csv
import sys
import fnmatch
import numpy as np
import subprocess
import argparse


average_fname = 'avg.csv'
average_worker_fname = 'avg-workers.csv'
comm_comp_fname = 'comm-comp.csv'
comm_comp_worker_fname = 'comm-comp-workers.csv'

parser = argparse.ArgumentParser(description='Generate a csv report from the input raw data.',
                                 usage='python extract.py -i [indir] -o [outfile]')

parser.add_argument('-o', '--outfile', type=str, default='sys.stdout',
                    choices=['sys.stdout', 'file'],
                    help='The file to save the report.'
                    ' Default: (indir)-report.csv')

parser.add_argument('-i', '--indir', type=str, default=None,
                    help='The directory containing the collected data.')

parser.add_argument('-r', '--report', type=str, default='all',
                    choices=['generate', 'collect', 'aggregate', 'all'],
                    help='The report type.')

parser.add_argument('-s', '--script', type=str, default='report_workers.py',
                    help='The path to the report_workers script.')


def generate_reports(input, report_script):
    records = []
    for dirs, subdirs, files in os.walk(input):
        if 'report' in subdirs:
            print(dirs)
            subprocess.call(['python', report_script, '-r', 'comm-comp',
                             '-i', os.path.join(dirs, 'report'),
                             '-o', os.path.join(dirs, comm_comp_worker_fname),
                             '-p', 'worker-*.csv'])

            subprocess.call(['python', report_script, '-r', 'avg',
                             '-i', os.path.join(dirs, 'report'),
                             '-o', os.path.join(dirs, average_worker_fname),
                             '-p', 'worker-*.csv'])


def write_avg(files, outfile):
    acc_data = []
    num = 0
    default_funcs = []
    default_header = []
    for f in files:
        data = np.genfromtxt(f, dtype=str, delimiter='\t')
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
    acc_data = np.around(acc_data/num, 2)
        
    writer = csv.writer(outfile, delimiter='\t')
    writer.writerow(default_header)
    for f, r in zip(default_funcs, acc_data):
        writer.writerow([f]+list(r))


def aggregate_reports(input):
    date_pattern = '*.*-*-*'
    for dirs, subdirs, _ in os.walk(input):
        sdirs = fnmatch.filter(subdirs, date_pattern)
        if len(sdirs) == 0:
            continue
        files = [os.path.join(dirs, s, comm_comp_worker_fname) for s in sdirs]
        write_avg(files, open(os.path.join(dirs, comm_comp_fname), 'w'))

        files = [os.path.join(dirs, s, average_worker_fname) for s in sdirs]
        write_avg(files, open(os.path.join(dirs, average_fname), 'w'))


def collect_reports(input, outfile, filename):
    # pass
    header = ['parts', 'slices', 'turns', 'n', 'omp', 'N']
    records = []
    for dirs, subdirs, files in os.walk(input):
        if filename not in files:
            continue

        print(dirs)
        config = dirs.split('/')[-1]
        ts = config.split('_t')[1].split('_')[0]
        ps = config.split('_p')[1].split('_')[0]
        ss = config.split('_s')[1].split('_')[0]
        ws = config.split('_w')[1].split('_')[0]
        oss = config.split('_o')[1].split('_')[0]
        Ns = config.split('_N')[1].split('_')[0]

        data = np.genfromtxt(os.path.join(dirs, filename),
                             dtype=str, delimiter='\t')

        data_head = data[0]
        data = data[1:]
        for r in data:
            records.append([ps, ss, ts, ws, oss, Ns] + list(r))

    records.sort(key=lambda a: (int(a[0]), int(a[1]), int(a[2]),
                                int(a[3]), int(a[4]), int(a[5])))
    writer = csv.writer(outfile, delimiter='\t')
    writer.writerow(header + list(data_head))
    writer.writerows(records)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.indir == None:
        print("You have to specify the input directory.")
        exit(-1)
    # if args.outfile == None:
    #     args.outfile = open(os.path.join(args.indir, 'report.csv'), 'w')
    #     args.outfile = open(os.path.join(args.indir, 'report.csv'), 'w')
    # elif args.outfile == 'sys.stdout':
    #     args.outfile = sys.stdout
    # print(args.outfile)
    if args.report == 'generate':
        generate_reports(args.indir, args.script)
    elif args.report == 'aggregate':
        aggregate_reports(args.indir)
    elif args.report == 'collect':
        if args.outfile == 'sys.stdout':
            collect_reports(args.indir, sys.stdout, average_fname)
            collect_reports(args.indir, sys.stdout, comm_comp_fname)
        elif args.outfile == 'file':
            collect_reports(args.indir, open(os.path.join(
                args.indir, 'avg-report.csv'), 'w'), average_fname)
            collect_reports(args.indir, open(os.path.join(
                args.indir, 'comm-comp-report.csv'), 'w'), comm_comp_fname)
    elif args.report == 'all':
        generate_reports(args.indir, args.script)
        aggregate_reports(args.indir)
        if args.outfile == 'sys.stdout':
            collect_reports(args.indir, sys.stdout, average_fname)
            collect_reports(args.indir, sys.stdout, comm_comp_fname)
        elif args.outfile == 'file':
            collect_reports(args.indir, open(os.path.join(
                args.indir, 'avg-report.csv'), 'w'), average_fname)
            collect_reports(args.indir, open(os.path.join(
                args.indir, 'comm-comp-report.csv'), 'w'), comm_comp_fname)