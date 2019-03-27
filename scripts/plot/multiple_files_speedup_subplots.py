import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.lines as mlines
import matplotlib.ticker
import sys
from plot.plotting_utilities import *

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)


plots_config = {
    # 'plot1': {
    #     'files': {
    #         res_dir+'raw/SPS-b72-4MPPB-t10k-mpich3/comm-comp-report.csv': {
    #             'key': 'sps-mpich3',
    #             'lines': {
    #                 'omp': ['10'],
    #                 'type': ['total']
    #             }
    #         },

    #         res_dir+'raw/SPS-b72-4MPPB-t10k-mvapich2/comm-comp-report.csv': {
    #             'key': 'sps-mvapich2',
    #             'lines': {
    #                 'omp': ['10'],
    #                 'type': ['total']
    #             }
    #         },
    #         res_dir+'raw/SPS-b72-4MPPB-t10k-openmpi3/comm-comp-report.csv': {
    #             'key': 'sps-openmpi3',
    #             'lines': {
    #                 'omp': ['10'],
    #                 'type': ['total']
    #             }
    #         },
    #         res_dir+'raw/LHC-96B-2MPPB-t10k-mpich3/comm-comp-report.csv': {
    #             'key': 'lhc-mpich3',
    #             'lines': {
    #                 'omp': ['10'],
    #                 'type': ['total']
    #             }
    #         },

    #         res_dir+'raw/LHC-96B-2MPPB-t10k-openmpi3/comm-comp-report.csv': {
    #             'key': 'lhc-openmpi3',
    #             'lines': {
    #                 'omp': ['10'],
    #                 'type': ['total']
    #             }
    #         },
    #         res_dir+'raw/LHC-96B-2MPPB-t10k-mvapich2/comm-comp-report.csv': {
    #             'key': 'lhc-mvapich2',
    #             'lines': {
    #                 'omp': ['10'],
    #                 'type': ['total']
    #             }
    #         },
    #         res_dir+'raw/PS-b21-t10k-mpich3/comm-comp-report.csv': {
    #             'key': 'ps-mpich3',
    #             'lines': {
    #                 'omp': ['10'],
    #                 'type': ['total']
    #             }
    #         },
    #         res_dir+'raw/PS-b21-t10k-openmpi3/comm-comp-report.csv': {
    #             'key': 'ps-openmpi3',
    #             'lines': {
    #                 'omp': ['10'],
    #                 'type': ['total']
    #             }
    #         },
    #         res_dir+'raw/PS-b21-t10k-mvapich2/comm-comp-report.csv': {
    #             'key': 'ps-mvapich2',
    #             'lines': {
    #                 'omp': ['10'],
    #                 'type': ['total']
    #             }
    #         },

    #     },
    #     'labels': {
    #         'lhc-mpich3': 'lhc-mpich3',
    #         'lhc-orig': 'lhc-orig',
    #         'lhc-openmpi3': 'lhc-openmpi3',
    #         'lhc-mvapich2': 'lhc-mvapich2',
    #         'sps-mpich3': 'sps-mpich3',
    #         'sps-orig': 'sps-orig',
    #         'sps-openmpi3': 'sps-openmpi3',
    #         'sps-mvapich2': 'sps-mvapich2',
    #         'ps-mpich3': 'ps-mpich3',
    #         'ps-orig': 'ps-orig',
    #         'ps-openmpi3': 'ps-openmpi3',
    #         'ps-mvapich2': 'ps-mvapich2',
    #     },
    #     'markers': {
    #         'lhc-mpich3': 'o',
    #         'lhc-orig': 'o',
    #         'lhc-openmpi3': 'o',
    #         'lhc-mvapich2': 'o',
    #         'sps-mpich3': 's',
    #         'sps-orig': 's',
    #         'sps-openmpi3': 's',
    #         'sps-mvapich2': 's',
    #         'ps-orig': 'x',
    #         'ps-mpich3': 'x',
    #         'ps-openmpi3': 'x',
    #         'ps-mvapich2': 'x',
    #     },
    #     'ls': {
    #         'lhc-orig': '-',
    #         'lhc-mpich3': '-',
    #         'lhc-openmpi3': '-',
    #         'lhc-mvapich2': '-',
    #         'sps-orig': ':',
    #         'sps-mpich3': ':',
    #         'sps-openmpi3': ':',
    #         'sps-mvapich2': ':',
    #         'ps-orig': '--',
    #         'ps-mpich3': '--',
    #         'ps-openmpi3': '--',
    #         'ps-mvapich2': '--',
    #     },
    #     'colors': {
    #         'lhc': 'tab:blue',
    #         'sps': 'tab:orange',
    #         'ps': 'tab:green',

    #         # 'mpich3': 'tab:blue',
    #         # 'openmpi3': 'tab:orange',
    #         # 'mvapich2': 'tab:green',
    #     },
    #     'hatches': {
    #         'mpich3': 'x',
    #         'openmpi3': '-',
    #         'mvapich2': '',

    #         # 'lhc': '/',
    #         # 'sps': '',
    #         # 'ps': 'x',
    #     },
    #     'reference': {
    #         'sps': {'time': 430., 'parts': 4000000, 'turns': 100},
    #         'lhc': {'time': 2120., 'parts': 2000000, 'turns': 1000},
    #         'ps': {'time': 1623.7, 'parts': 4000000, 'turns': 2000},
    #     },

    #     # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    #     'x_name': 'n',
    #     'omp_name': 'omp',
    #     'y_name': 'avg_time(sec)',
    #     # 'y_err_name': 'std',
    #     'xlabel': 'Cores (x10)',
    #     'ylabel': 'Speedup',
    #     'title': 'Alternative MPI Versions',
    #     # 'ylim': {
    #     #     'speedup': [0, 210]
    #     # },
    #     # 'nticks': 6,
    #     'legend_loc': 'upper left',
    #     'figsize': (6, 4),
    #     'image_name': images_dir + 'mpi-versions-bars-1.pdf'

    # },

    'plot1': {
        'files': {
            # res_dir+'raw/SPS-b72-4MPPB-t10k-mpich3/comm-comp-report.csv': {
            #     'key': 'sps-mpich3',
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },

            # res_dir+'raw/SPS-b72-4MPPB-t10k-mvapich2/comm-comp-report.csv': {
            #     'key': 'sps-mvapich2',
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },
            # res_dir+'raw/SPS-b72-4MPPB-t10k-openmpi3/comm-comp-report.csv': {
            #     'key': 'sps-openmpi3',
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },
            res_dir+'raw/LHC-96B-2MPPB-t10k-mpich3/comm-comp-report.csv': {
                'key': 'lhc-mpich3',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },

            # res_dir+'raw/LHC-96B-2MPPB-t10k-openmpi3/comm-comp-report.csv': {
            #     'key': 'lhc-openmpi3',
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },
            # res_dir+'raw/LHC-96B-2MPPB-t10k-mvapich2/comm-comp-report.csv': {
            #     'key': 'lhc-mvapich2',
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },
            res_dir+'raw/LHC-lb-mpich3/comm-comp-report.csv': {
                'key': 'lhc-lbmpich3',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },

            # res_dir+'raw/PS-b21-t10k-mpich3/comm-comp-report.csv': {
            #     'key': 'ps-mpich3',
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },
            # res_dir+'raw/PS-b21-t10k-openmpi3/comm-comp-report.csv': {
            #     'key': 'ps-openmpi3',
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },
            # res_dir+'raw/PS-b21-t10k-mvapich2/comm-comp-report.csv': {
            #     'key': 'ps-mvapich2',
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },

        },
        'labels': {
            'lhc-mpich3': 'lhc-mpich3',
            'lhc-lbmpich3': 'lhc-lb-mpich3',
            'lhc-orig': 'lhc-orig',
            'lhc-openmpi3': 'lhc-openmpi3',
            'lhc-mvapich2': 'lhc-mvapich2',
            'sps-mpich3': 'sps-mpich3',
            'sps-orig': 'sps-orig',
            'sps-openmpi3': 'sps-openmpi3',
            'sps-mvapich2': 'sps-mvapich2',
            'ps-mpich3': 'ps-mpich3',
            'ps-orig': 'ps-orig',
            'ps-openmpi3': 'ps-openmpi3',
            'ps-mvapich2': 'ps-mvapich2',
        },
        'markers': {
            'lhc-lbmpich3': 'o',
            'lhc-mpich3': 'o',
            'lhc-orig': 'o',
            'lhc-openmpi3': 'o',
            'lhc-mvapich2': 'o',
            'sps-mpich3': 's',
            'sps-orig': 's',
            'sps-openmpi3': 's',
            'sps-mvapich2': 's',
            'ps-orig': 'x',
            'ps-mpich3': 'x',
            'ps-openmpi3': 'x',
            'ps-mvapich2': 'x',
        },
        'ls': {
            'lhc-orig': '-',
            'lhc-lbmpich3': '-',
            'lhc-mpich3': '-',
            'lhc-openmpi3': '-',
            'lhc-mvapich2': '-',
            'sps-orig': ':',
            'sps-mpich3': ':',
            'sps-openmpi3': ':',
            'sps-mvapich2': ':',
            'ps-orig': '--',
            'ps-mpich3': '--',
            'ps-openmpi3': '--',
            'ps-mvapich2': '--',
        },
        'colors': {
            'lhc': 'tab:blue',
            'sps': 'tab:orange',
            'ps': 'tab:green',

            # 'mpich3': 'tab:blue',
            # 'openmpi3': 'tab:orange',
            # 'mvapich2': 'tab:green',
        },
        'hatches': {
            'mpich3': 'x',
            'lbmpich3': 'o',
            'openmpi3': '-',
            'mvapich2': '',
            # 'lhc': '/',
            # 'sps': '',
            # 'ps': 'x',
        },
        'reference': {
            'sps': {'time': 430., 'parts': 4000000, 'turns': 100},
            'lhc': {'time': 2120., 'parts': 2000000, 'turns': 1000},
            'ps': {'time': 1623.7, 'parts': 4000000, 'turns': 2000},
        },

        # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
        'x_name': 'n',
        'omp_name': 'omp',
        'y_name': 'avg_time(sec)',
        # 'y_err_name': 'std',
        'xlabel': 'Cores (x10)',
        'ylabel': 'Speedup',
        'title': 'Alternative MPI Versions',
        # 'ylim': {
        #     'speedup': [0, 210]
        # },
        # 'nticks': 6,
        'legend_loc': 'upper left',
        'figsize': (6, 4),
        'image_name': images_dir + 'mpi-ver-lb-bars-1.pdf'

    },

}

if __name__ == '__main__':
    for plot_key, config in plots_config.items():
        plots_dir = {}
        for file in config['files'].keys():
            # print(file)
            data = np.genfromtxt(file, delimiter='\t', dtype=str)
            header, data = list(data[0]), data[1:]
            temp = get_plots(header, data, config['files'][file]['lines'],
                             exclude=config['files'][file].get('exclude', []))
            temp[config['files'][file]['key']] = temp['10-total']
            del temp['10-total']
            plots_dir.update(temp)

        # print(plots_dir)
        fig = plt.figure(figsize=config['figsize'])
        # ax1 = fig.add_subplot(111)
        # ax2 = ax1.twinx()

        plt.grid(True, which='major', alpha=0.5)
        plt.grid(False, which='major', axis='x')
        # plt.minorticks_on()
        plt.title(config['title'])
        # ax1.set_title(config['title'])
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        # plt.ylim(config['ylim']['speedup'])

        # plt.yscale('log', basex=2)
        # if 'ylim' in config:
        #     plt.ylim(config['ylim'])

        pos = 0
        # width = 0.1
        step = 0.25
        width = 2. / (len(plots_dir.keys())+1)
        for case in ['lhc', 'sps', 'ps']:
            for mpiv in ['mpich3', 'openmpi3', 'mvapich2', 'lbmpich3']:
                key = '{}-{}'.format(case, mpiv)
                if key not in plots_dir:
                    continue
                values = plots_dir[key]

                label = config['labels'][key]

                x = np.array(values[:, header.index(config['x_name'])], float)
                omp = np.array(
                    values[:, header.index(config['omp_name'])], float)
                x = (x) * omp

                y = np.array(values[:, header.index(config['y_name'])], float)
                parts = np.array(values[:, header.index('parts')], float)
                turns = np.array(values[:, header.index('turns')], float)
                # This is the throughput
                y = parts * turns / y

                # Now the reference, 1thread
                yref = config['reference'][case]['time']
                partsref = config['reference'][case]['parts']
                turnsref = config['reference'][case]['turns']
                yref = partsref * turnsref / yref

                speedup = y / yref

                # efficiency = 100 * speedup / x
                plt.bar(x//10 + pos, speedup, width=width, 
                    # color=[config['colors'][case]],
                    color='0.6',
                    label=label,
                    edgecolor=config['colors'][case],
                    alpha=0.8,  
                    hatch=config['hatches'][mpiv])
                pos += width
                # We want speedup, compared to 1 worker with 1 thread
                # plt.errorbar(x//10, speedup, yerr=None, color=config['colors'][key],
                #              capsize=2, marker=config['markers'][key],
                #              markersize=4,
                #              linewidth=2., label=label,
                #              ls=config['ls'][key])
            pos+=width * step
            # if '10' in key:
            #     plt.xticks(x//10)
            # annotate_max(plt.gca(), x//10, speedup, ha='center', va='bottom',
            #              size='10')

            # ax2.errorbar(x//10, efficiency, yerr=None, color=config['colors']['efficiency'],
            #              capsize=2, marker=config['markers'][key], markersize=4,
            #              linewidth=1.)

        if 'extra' in config:
            for c in config['extra']:
                exec(c)

        # nticks = config['nticks']
        # plt.gca().yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
        # ax2.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
        # for tl in ax1.get_yticklabels():
        #     tl.set_color(config['colors']['speedup'])

        # handles = []
        # for k, v in config['markers'].items():
        #     line = mlines.Line2D([], [], color='black',
        #                          marker=v, label=config['labels'][k])
        #     handles.append(line)
        plt.xticks(x//10 + pos/2, np.array(x//10, int))

        plt.legend(loc=config['legend_loc'], fancybox=True, fontsize=10.5,
                   labelspacing=0, borderpad=0.5, framealpha=0.8,
                   handletextpad=0.5, handlelength=2, borderaxespad=0)
        plt.tight_layout()
        save_and_crop(fig, config['image_name'], dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
