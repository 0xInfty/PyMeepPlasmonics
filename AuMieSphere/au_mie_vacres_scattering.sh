#!/bin/bash

#thgit
#cd AuMieSphere
mpirun --use-hwthread-cpus -np 6 python p_au_mie_scattering.py -s "Max80Res1" -f "AuMieSphere/AuMie/10)MaxRes/Max80Res" -r 4 -res 1
mpirun --use-hwthread-cpus -np 6 python p_au_mie_scattering.py -s "Max80Res2" -f "AuMieSphere/AuMie/10)MaxRes/Max80Res" -r 4 -res 2
mpirun --use-hwthread-cpus -np 6 python p_au_mie_scattering.py -s "Max80Res3" -f "AuMieSphere/AuMie/10)MaxRes/Max80Res" -r 4 -res 3
#mpirun --use-hwthread-cpus -np 6 python p_au_mie_scattering.py -s "Max80Res4" -f "AuMieSphere/AuMie/10)MaxRes/Max80Res" -r 4 -res 4
#mpirun --use-hwthread-cpus -np 6 python p_au_mie_scattering.py -s "Max80Res5" -f "AuMieSphere/AuMie/10)MaxRes/Max80Res" -r 4 -res 5
