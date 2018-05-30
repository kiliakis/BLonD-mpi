""" A different implementation of task-pull with less communication and full
use of resources (mainly-idle parent shares with worker). Sentinels are used in
place of tags. Start parent with 'python <filename.py>' rather than mpirun;
parent will then spawn specified number of workers. Work is randomized to
demonstrate dynamic allocation. Worker logs are collectively passed back to
parent at the end in place of results. Comments and output are both
deliberately excessive for instructional purposes. """

from mpi4py import MPI
import random
import time
import sys
from task_pull_worker_10 import worker

# n_workers = 9
n_tasks = 50
# worker = sys.path[0]+'/10-task-pull-worker.py'

# Parent
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.rank
    n_workers = comm.size - 1



    # print(worker)
    # sys.exit()
    # sys.stdout.flush()
    if rank != 0:
        worker()
        exit(0)

    newcomm = comm.Split(rank==0, rank)
    # print('newcomm')
    # print(newcomm)
    # print(newcomm.size)
    # print(newcomm.Get_group())

    intercomm = newcomm.Create_intercomm(0, MPI.COMM_WORLD, 1)
    # print('intercomm')
    # print(intercomm)
    # print(intercomm.size)
    # print(intercomm.Get_group())
    comm = intercomm
    n_workers = comm.Get_remote_size()
    print('Master[%d]: Hostname: %s' % (rank, MPI.Get_processor_name()))
    # Start clock
    start = MPI.Wtime()

    # Random 1-9s tasks
    task_list = [random.randint(1, 9) for task in range(n_tasks)]
    total_time = sum(task_list)

    # Append stop sentinel for each worker
    msg_list = task_list + ([StopIteration] * n_workers)

    # Spawn workers
    # comm = MPI.COMM_WORLD.Spawn(
    #     sys.executable,
    #     args=[worker],
    #     maxprocs=n_workers)

    # Reply to whoever asks until done
    status = MPI.Status()
    for position, msg in enumerate(msg_list):
        comm.recv(source=MPI.ANY_SOURCE, status=status)
        comm.send(obj=msg, dest=status.Get_source())

        # Simple (loop position) progress bar
        percent = ((position + 1) * 100) // (n_tasks + n_workers)
        sys.stdout.write(
            '\rProgress: [%-50s] %3i%% ' %
            ('=' * (percent // 2), percent))
        sys.stdout.flush()

    # Gather reports from workers
    reports = []
    reports = comm.gather(reports, root=MPI.ROOT)

    # Print summary
    workers = 0
    tasks = 0
    time = 0
    print('\n\n  Worker   Tasks    Time')
    print('-' * 26)
    for worker, report in enumerate(reports):
        print('%8i%8i%8i' % (worker, len(report), sum(report)))
        workers += 1
        tasks += len(report)
        time += sum(report)
    print('-' * 26)
    print('%8i%8i%8i' % (workers, tasks, time))

    # Check all in order
    assert workers == n_workers, 'Missing workers'
    assert tasks == n_tasks, 'Lost tasks'
    assert time == total_time, 'Output != assigned input'

    # Final statistics
    finish = MPI.Wtime() - start
    efficiency = (total_time * 100.) / (finish * n_workers)
    print('\nProcessed in %.2f secs' % finish)
    print('%.2f%% efficient' % efficiency)

    # Shutdown
    comm.Disconnect()
