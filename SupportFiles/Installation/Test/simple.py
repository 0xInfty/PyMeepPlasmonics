import h5py as h5
from mpi4py import MPI

#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
rank = MPI.COMM_WORLD.rank

if rank==0: print(f"Got to load modules and start instance of mpi4py with rank {rank}")

#f = h5.File("/nfs/home/vpais/PyMeepResults/TestParallel.h5", "w", driver="mpio", comm=comm)
f = h5.File("/nfs/home/vpais/PyMeepResults/TestParallel.h5", "w", driver="mpio", comm=MPI.COMM_WORLD)

if rank==0: print("Got to open file")

f["hey"] = [2021,9,19,11,50,0]

if rank==0: print("Got to save data in new dataset hey")

f.close()

if rank==0: print("Got to close file")
