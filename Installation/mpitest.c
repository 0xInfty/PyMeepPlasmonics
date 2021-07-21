/*
 * Usage:
 * mpicc mpitest.c -o mpitest
 * mpirun -np 2 -H `hostname` ./mpitest
 */

#include <stdio.h>
#include <mpi.h>

int  main(int argc, char *argv[])
{
    int rank, size, h_len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);

    // get rank of this proces
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // get total process number
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Get_processor_name(hostname, &h_len);
    printf("Start! rank:%d size: %d at %s\n", rank, size,hostname);
    //do something
    printf("Done!  rank:%d size: %d at %s\n", rank, size,hostname);

    MPI_Finalize();
    return 0;
}
