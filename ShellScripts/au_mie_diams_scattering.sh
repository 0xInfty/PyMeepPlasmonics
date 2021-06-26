#!/bin/bash

mpirun --use-hwthread-cpus -np 6 python -m mpi4py _au_mie_scattering.py -np 6 --parallel True -res 4 --from-um-factor 10e-3 -s "Diam48FU10Res4" -f "AuMieSphere/AuMie/7)Diameters/WLen4560" -r 24 -pp "R" --wlen-range "np.array([450,600])" 
mpirun --use-hwthread-cpus -np 6 python -m mpi4py u_au_mie_scattering.py -np 6 --parallel True -res 4 --from-um-factor 10e-3 -s "Diam64FU10Res4" -f "AuMieSphere/AuMie/7)Diameters/WLen4560" -r 32 -pp "R" --wlen-range "np.array([450,600])"
mpirun --use-hwthread-cpus -np 6 python -m mpi4py u_au_mie_scattering.py -np 6 --parallel True -res 4 --from-um-factor 10e-3 -s "Diam80FU10Res4" -f "AuMieSphere/AuMie/7)Diameters/WLen4560" -r 40 -pp "R" --wlen-range "np.array([450,600])"
mpirun --use-hwthread-cpus -np 6 python -m mpi4py u_au_mie_scattering.py -np 6 --parallel True -res 4 --from-um-factor 10e-3 -s "Diam103FU10Res4" -f "AuMieSphere/AuMie/7)Diameters/WLen4560" -r 51.5 -pp "R" --wlen-range "np.array([450,600])"
