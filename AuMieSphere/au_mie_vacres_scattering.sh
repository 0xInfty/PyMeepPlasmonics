#!/bin/bash

mpirun --use-hwthread-cpus -np 6 python p_au_mie_scattering.py -s "Max80FU20Res1" -f "AuMieSphere/AuMie/10)MaxRes/Max80FU20Res" --from-um-factor 20e-3 -res 1 -r 2 --wlen-range "[22.5, 30]"
mpirun --use-hwthread-cpus -np 6 python p_au_mie_scattering.py -s "Max80FU20Res2" -f "AuMieSphere/AuMie/10)MaxRes/Max80FU20Res" --from-um-factor 20e-3 -res 2 -r 2 --wlen-range "[22.5, 30]"
mpirun --use-hwthread-cpus -np 6 python p_au_mie_scattering.py -s "Max80FU20Res3" -f "AuMieSphere/AuMie/10)MaxRes/Max80FU20Res" --from-um-factor 20e-3 -res 3 -r 2 --wlen-range "[22.5, 30]"
mpirun --use-hwthread-cpus -np 6 python p_au_mie_scattering.py -s "Max80FU20Res4" -f "AuMieSphere/AuMie/10)MaxRes/Max80FU20Res" --from-um-factor 20e-3 -res 4 -r 2 --wlen-range "[22.5, 30]"
mpirun --use-hwthread-cpus -np 6 python p_au_mie_scattering.py -s "Max80FU20Res5" -f "AuMieSphere/AuMie/10)MaxRes/Max80FU20Res" --from-um-factor 20e-3 -res 5 -r 2 --wlen-range "[22.5, 30]"
mpirun --use-hwthread-cpus -np 6 python p_au_mie_scattering.py -s "Max80FU20Res6" -f "AuMieSphere/AuMie/10)MaxRes/Max80FU20Res" --from-um-factor 20e-3 -res 6 -r 2 --wlen-range "[22.5, 30]"
mpirun --use-hwthread-cpus -np 6 python p_au_mie_scattering.py -s "Max80FU20Res7" -f "AuMieSphere/AuMie/10)MaxRes/Max80FU20Res" --from-um-factor 20e-3 -res 7 -r 2 --wlen-range "[22.5, 30]"
mpirun --use-hwthread-cpus -np 6 python p_au_mie_scattering.py -s "Max80FU20Res8" -f "AuMieSphere/AuMie/10)MaxRes/Max80FU20Res" --from-um-factor 20e-3 -res 8 -r 2 --wlen-range "[22.5, 30]"
mpirun --use-hwthread-cpus -np 6 python p_au_mie_scattering.py -s "Max80FU20Res9" -f "AuMieSphere/AuMie/10)MaxRes/Max80FU20Res" --from-um-factor 20e-3 -res 9 -r 2 --wlen-range "[22.5, 30]"
#mpirun --use-hwthread-cpus -np 6 python p_au_mie_scattering.py -s "Max80FU20Res10" -f "AuMieSphere/AuMie/10)MaxRes/Max80FU20Res" --from-um-factor 20e-3 -res 10 -r 2 --wlen-range "[22.5, 30]"
#mpirun --use-hwthread-cpus -np 6 python p_au_mie_scattering.py -s "Max80FU20Res11" -f "AuMieSphere/AuMie/10)MaxRes/Max80FU20Res" --from-um-factor 20e-3 -res 11 -r 2 --wlen-range "[22.5, 30]"
#mpirun --use-hwthread-cpus -np 6 python p_au_mie_scattering.py -s "Max80FU20Res12" -f "AuMieSphere/AuMie/10)MaxRes/Max80FU20Res" --from-um-factor 20e-3 -res 12 -r 2 --wlen-range "[22.5, 30]"
