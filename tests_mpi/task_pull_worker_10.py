""" A different implementation of task-pull with less communication and full
use of resources (mainly-idle parent shares with worker). Sentinels are used in
place of tags. Start parent with 'python <filename.py>' rather than mpirun;
parent will then spawn specified number of workers. Work is randomized to
demonstrate dynamic allocation. Worker logs are collectively passed back to
parent at the end in place of results. Comments and output are both
deliberately excessive for instructional purposes. """
from mpi4py import MPI
import time


def worker():
# if __name__ == '__main__':
    # Connect to parent
    # try:
    #     comm = MPI.Comm.Get_parent()
    #     rank = comm.Get_rank()
    # except:
    #     raise ValueError('Could not connect to parent - ' + usage)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    newcomm = comm.Split(rank==0, rank)
    # print('newcomm')
    # print(newcomm)
    # print(newcomm.size)
    # print(newcomm.Get_group())

    intercomm = newcomm.Create_intercomm(0, MPI.COMM_WORLD, 0)
    # print('intercomm')
    # print(intercomm)
    # print(intercomm.size)
    # print(intercomm.Get_group())
    comm = intercomm
    print('Worker[%d]: Hostname: %s' % (rank, MPI.Get_processor_name()))
    # Ask for work until stop sentinel
    log = []
    for task in iter(lambda: comm.sendrecv(None, dest=0), StopIteration):
        log.append(task)

        # Do work (or not!)
        time.sleep(task)

    # Collective report to parent
    comm.gather(sendobj=log, root=0)

    # Shutdown
    # comm.Disconnect()
    exit(0)
    