#!/usr/bin/python
import os
import numpy as np
import sys
import fnmatch
import csv
import argparse
import re

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'


parser = argparse.ArgumentParser(description='Report the particles, time and latency assigned to each worker.',
                                 usage='python script.py [-p file_pattern] [-i indir] [-o outfile]')

parser.add_argument('-p', '--pattern', type=str, default='worker-*.log',
                    help='The report file names pattern. '
                    ' Default: worker-*.log')

parser.add_argument('-o', '--outfile', type=argparse.FileType('w'),
                    default=sys.stdout,
                    help='The file(s) to save the report.'
                    ' Default: Print to the stdout')

parser.add_argument('-i', '--indir', type=str, default='./',
                    help='The directory containing the report files.'
                    ' Default: Use the current working directory.')


parser.add_argument('-r', '--report', type=str, choices=['particles'],
                    default='particles',
                    help='Choose from 1 report types: particles.'
                    ' Default: particles.')


def report_particles(indir, files, outfile):
    re_tracking = re.compile('.*\[(\d+)\].*Tracking\s+(\d+)\s+particles.') 
    re_time = re.compile('.*\[(\d+)\].*Time\s(.*)\ssec.')
    re_latency = re.compile('.*\[(\d+)\].*Latency\s(.*)\ssec/particle.')
    data = {}
    for f in files:
        for line in open(indir + '/' + f, 'r'):
            if (re_tracking.search(line)):
                match = re_tracking.search(line)
                wid, parts = match.groups()
                wid = int(wid)
                if wid not in data:
                    data[wid] = {}
                if 'parts' not in data[wid]:
                    data[wid]['parts']=[]
                data[wid]['parts'].append((parts))
            elif (re_time.search(line)):
                match = re_time.search(line)
                wid, time = match.groups()
                wid = int(wid)
                if wid not in data:
                    data[wid] = {}
                if 'time' not in data[wid]:
                    data[wid]['time']=[]
                data[wid]['time'].append((time))

            elif (re_latency.search(line)):
                match = re_latency.search(line)
                wid, latency = match.groups()
                wid = int(wid)
                if wid not in data:
                    data[wid] = {}
                if 'latency' not in data[wid]:
                    data[wid]['latency']=[]
                data[wid]['latency'].append((latency))

    outfile.write('wid\tparts\ttimes\tlatencies\n')
    for wid in sorted(data.keys()):
        parts = '|'.join(data[wid]['parts'])
        times = '|'.join(data[wid]['time'])
        latencies = '|'.join(data[wid]['latency'])
        outfile.write('{}\t{}\t{}\t{}\n'.format(wid, parts, times, latencies))
    outfile.close()

if __name__ == '__main__':
    args = parser.parse_args()
    file_pattern = args.pattern
    indir = args.indir
    files = fnmatch.filter(os.listdir(indir), file_pattern)
    outfile = args.outfile
    if outfile == 'sys.stdout':
        outfile = sys.stdout

    if args.report == 'particles':
        report_particles(indir, files, outfile)
