import sys
from mpi4py import MPI
import numpy as np

mpi_type = {
    'd': MPI.DOUBLE,
    'i': MPI.INT
}

executable = 'python'
n_workers = 2
worker_script = sys.path[0] + '/worker.py'


def multi_scatter_master(comm, vars):

    var_list = [(k, v[1]) for k, v in vars.items()]
    comm.bcast(var_list, root=MPI.ROOT)

    for (name, dtype) in var_list:
        val = vars[name][0]
        # calculate counts and displs
        basesize = len(val) // n_workers
        plusone = len(val) - basesize * n_workers
        counts = np.array([basesize+1]*plusone + [basesize] *
                          (n_workers-plusone), dtype='i')
        displs = np.append([0], np.cumsum(counts[:-1]))

        recvbuf = np.array(0, dtype='i')
        comm.Scatter(counts, recvbuf, root=MPI.ROOT)

        recvbuf = np.empty(1, dtype=dtype)
        comm.Scatterv([val, counts, displs, mpi_type[dtype]],
                            recvbuf, root=MPI.ROOT)
    # comm.Barrier()


def multi_scatter_worker(comm):
    var_list = None
    var_list = comm.bcast(var_list, root=0)
    vals = {}
    sendbuf = None
    for name, dtype in var_list:
        count = np.array(0, dtype='i')
        comm.Scatter(sendbuf, count, root=0)
        # print('Worker[%d]: Received buffer: ' % (rank), count)
        recvbuf = np.empty(count, dtype=dtype)
        comm.Scatterv(sendbuf, recvbuf, root=0)
        vals[name] = recvbuf
        # print('Worker[%d]: Received buffer: ' % (rank), recvbuf)
    return vals
    # comm.Barrier()



def multi_bcast_master(comm, vars):
    comm.bcast(vars, root=MPI.ROOT)


def multi_bcast_worker(comm):
    vars = None
    vars = comm.bcast(vars, root=0)
    return vars