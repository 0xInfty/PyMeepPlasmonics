#!/bin/bash

mpirun --use-hwthread-cpus -np 6 python -m mpi4py n_au_mie_scattering.py -np 6 --parallel True -res 4 --from-um-factor 10e-3 -s "TestPaperJCFitDiam48" -f "AuMieSphere/AuMie/13)TestPaper/4)PaperJCFit/TestPaperJCFitDiams" -r 24 -pp "JC" --wlen-range "np.array([450,600])" 
mpirun --use-hwthread-cpus -np 6 python -m mpi4py n_au_mie_scattering.py -np 6 --parallel True -res 4 --from-um-factor 10e-3 -s "TestPaperJCFitDiam64" -f "AuMieSphere/AuMie/13)TestPaper/4)PaperJCFit/TestPaperJCFitDiams" -r 32 -pp "JC" --wlen-range "np.array([450,600])"
mpirun --use-hwthread-cpus -np 6 python -m mpi4py n_au_mie_scattering.py -np 6 --parallel True -res 4 --from-um-factor 10e-3 -s "TestPaperJCFitDiam80" -f "AuMieSphere/AuMie/13)TestPaper/4)PaperJCFit/TestPaperJCFitDiams" -r 40 -pp "JC" --wlen-range "np.array([450,600])"
mpirun --use-hwthread-cpus -np 6 python -m mpi4py n_au_mie_scattering.py -np 6 --parallel True -res 4 --from-um-factor 10e-3 -s "TestPaperJCFitDiam103" -f "AuMieSphere/AuMie/13)TestPaper/4)PaperJCFit/TestPaperJCFitDiams" -r 51.5 -pp "JC" --wlen-range "np.array([450,600])"
