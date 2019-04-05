#!/usr/bin/python
import os
import csv
import sys
import fnmatch
import numpy as np
import subprocess
import argparse


this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
# average_fname = 'avg.csv'
# average_std_fname = 'avg-std.csv'
# average_worker_fname = 'avg-workers.csv'
# comm_comp_fname = 'comm-comp.csv'
# comm_comp_std_fname = 'comm-comp-std.csv'
log_fname = 'particles.csv'
log_fname_std = 'particles-std.csv'
log_worker_fname = 'particles-workers.csv'
worker_pattern = 'worker-*.log'

parser = argparse.ArgumentParser(description='Generate a csv report from the input raw data.',
                                 usage='python redistribute.py -i [indir] -o [outfile]')

parser.add_argument('-o', '--outfile', type=str, default='file',
                    choices=['sys.stdout', 'file'],
                    help='The file to save the report.'
                    ' Default: (indir)-report.csv')

parser.add_argument('-i', '--indir', type=str, default=None,
                    help='The directory containing the collected data.')

parser.add_argument('-r', '--report', type=str, default='all',
                    choices=['generate', 'collect', 'aggregate', 'all'],
                    help='The report type.')

parser.add_argument('-s', '--script', type=str, default=this_directory + 'helper_redistribute.py',
                    help='The path to the helper_redistribute script.')

parser.add_argument('-k', '--keep', type=int, default=2,
                    help='The number of top best runs to keep for the average calculation. Use -1 for all.')

parser.add_argument('-u', '--update', action='store_true',
                    help='Force update of already calculated reports.')


def generate_reports(input, report_script):
    print('\n--------Generating reports-------\n')
    records = []
    for dirs, subdirs, files in os.walk(input):
        if 'log' not in subdirs:
            continue
        ps = []
        print(dirs)
        log_dir = os.path.join(dirs, 'log')
        outfile = os.path.join(dirs, log_worker_fname)
        if (args.update or (not os.path.isfile(outfile))):
            ps.append(subprocess.Popen(['python', report_script, '-r', 'particles',
                                        '-i', log_dir, '-o', outfile,
                                        '-p', worker_pattern]))
        for p in ps:
            p.wait()


def calc_histo(files, outfile, outfile_std):
    acc_data = []
    default_header = []
    data_dic = {}
    for f in files:
        data = np.genfromtxt(f, dtype=str, delimiter='\t')
        header, data = data[0], data[1:]
        wids, data = data[:, 0], data[:, 1:]
        header = header[1:] # remove the wid

        if len(default_header) == 0:
            default_header = header
        elif not np.array_equal(default_header, header):
            print('Problem with file: ', indir+'/'+f)
            continue
        dic = {}
        ppb = int(f.split('_p')[1].split('_')[0])
        bunches = int(f.split('_b')[1].split('_')[0])
        workers = int(f.split('_w')[1].split('_')[0])
        parts_t0 = ppb * bunches // workers
        for i, h in enumerate(header):
            lst = []
            for d in data:
                if h == 'parts':
                    lst.append([parts_t0] + d[i].split('|'))
                else:
                    lst.append(d[i].split('|'))
            min_num = np.min([len(l) for l in lst])
            for j,l in enumerate(lst):
                lst[j] = l[:min_num]
            dic[h] = np.array(lst, float)
            if h == 'parts':
                dic[h] = np.abs(np.diff(dic[h]))/parts_t0
        for k, v in dic.items():
            # dic[k] = 100 * np.std(v, axis=0) / np.mean(v, axis=0)
            if k not in data_dic:
                data_dic[k] = []
            mean = np.mean(v, axis=0)
            std = np.std(v, axis=0)
            lst = []
            for m,s in zip(mean, std):
                if m == 0:
                    lst.append(0)
                else:
                    lst.append(100 * s/m)
            data_dic[k].append(lst)
            # if mean == 0 or std ==0:
            #     data_dic[k].append(0)
            # else:
                # data_dic[k].append(100 * np.std(v, axis=0) / np.mean(v, axis=0))


    #     for i, f in enumerate(funcs):
    #         if f not in data_dic:
    #             data_dic[f] = []
    #         data_dic[f].append(data[i])

    acc_data = []
    acc_data_std = []
    # sortid = [i[0]for i in sorted(enumerate(data_dic[funcs[-1]]),
    #                               key=lambda a:a[1][0])]
    for k, v in data_dic.items():
        # data_dic[k] = np.array(v)[sortid][:args.keep]
        acc_data.append([k] + list(np.around(np.mean(data_dic[k], axis=0), 2)))
        acc_data_std.append(
            [k] + list(np.around(np.std(data_dic[k], axis=0), 2)))

    # Transpose list of lists magic
    acc_data = list(map(list, zip(*acc_data)))
    writer1 = csv.writer(outfile, delimiter='\t')
    writer1.writerows(acc_data)
    writer2 = csv.writer(outfile_std, delimiter='\t')
    writer2.writerows(acc_data_std)


def aggregate_reports(input):
    print('\n--------Aggregating reports-------\n')

    date_pattern = '*.*-*-*'
    for dirs, subdirs, _ in os.walk(input):
        sdirs = fnmatch.filter(subdirs, date_pattern)
        if len(sdirs) == 0:
            continue
        files = [os.path.join(dirs, s, log_worker_fname) for s in sdirs]
        # print(files)
        try:
            calc_histo(files, open(os.path.join(dirs, log_fname), 'w'),
                open(os.path.join(dirs, log_fname_std), 'w'))
        except Exception as e:
            print('[Error] Dir: {}, Exception: {}'.format(dirs, e))


def collect_reports(input, outfile, filename):
    print('\n--------Collecting reports-------\n')
    header = ['parts', 'bunches', 'slices', 'turns', 'n', 'omp', 'N', 'red']
    records = []
    for dirs, subdirs, files in os.walk(input):
        if filename not in files:
            continue

        print(dirs)
        try:
            config = dirs.split('/')[-1]
            ts = config.split('_t')[1].split('_')[0]
            ps = config.split('_p')[1].split('_')[0]
            bs = config.split('_b')[1].split('_')[0]
            ss = config.split('_s')[1].split('_')[0]
            ws = config.split('_w')[1].split('_')[0]
            oss = config.split('_o')[1].split('_')[0]
            Ns = config.split('_N')[1].split('_')[0]
            rs = config.split('_r')[1].split('_')[0]

            data = np.genfromtxt(os.path.join(dirs, filename),
                                 dtype=str, delimiter='\t')

            data_head, data = data[0], data[1:]
            for r in data:
                records.append([ps, bs, ss, ts, ws, oss, Ns, rs] + list(r))
        except:
            print('[Error] dir ', dirs)
            continue
    records.sort(key=lambda a: (float(a[0]), int(a[1]), int(a[2]),
                                int(a[3]), int(a[4]), int(a[5]), int(a[6])))
    writer = csv.writer(outfile, delimiter='\t')
    writer.writerow(header + ['turn#', 'dp_std', 't_std', 'lat_std'])
    writer.writerows(records)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.indir == None:
        print("You have to specify the input directory.")
        exit(-1)

    if args.report in ['generate', 'all']:
        generate_reports(args.indir, args.script)
    if args.report in ['aggregate', 'all']:
        aggregate_reports(args.indir)
    # if args.report in ['collect', 'all']:
    #     if args.outfile == 'sys.stdout':
    #         collect_reports(args.indir, sys.stdout, average_fname)
    #         collect_reports(args.indir, sys.stdout, comm_comp_fname)
    #     elif args.outfile == 'file':
    #         collect_reports(args.indir,
    #                         open(os.path.join(args.indir, 'avg-std-report.csv'), 'w'),
    #                         average_std_fname)
    #         collect_reports(args.indir,
    #                         open(os.path.join(args.indir, 'avg-report.csv'), 'w'),
    #                         average_fname)
    #         collect_reports(args.indir,
    #                         open(os.path.join(args.indir,
    #                                           'comm-comp-report.csv'), 'w'),
    #                         comm_comp_fname)
    #         collect_reports(args.indir,
    #                         open(os.path.join(args.indir,
    #                                           'comm-comp-std-report.csv'), 'w'),
    #                         comm_comp_std_fname)
