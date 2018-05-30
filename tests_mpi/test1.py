from mpi4py import MPI
import numpy as np
from parutils import pprint
import sys

comm = MPI.COMM_WORLD
start = MPI.Wtime()

print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
sys.stdout.flush()
comm.Barrier()   # wait for everybody to synchronize _here_


pprint("-"*78)
pprint(" Running on %d cores" % comm.size)
pprint("-"*78)
sys.stdout.flush()

end = MPI.Wtime()
pprint("Total time: %f" % (end-start))
sys.stdout.flush()

comm.Barrier()

# Prepare a vector of N=5 elements to be broadcasted...
N = 4
if comm.rank == 0:
    A = np.arange(N, dtype=np.float64)    # rank 0 has proper data
else:
    A = np.empty(N, dtype=np.float64)     # all other just an empty array

# Broadcast A from rank 0 to everybody
comm.Bcast( [A, MPI.DOUBLE] )

# Everybody should now have the same...
print("[%02d] %s" % (comm.rank, A))