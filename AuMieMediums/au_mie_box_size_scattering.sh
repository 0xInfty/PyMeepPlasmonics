#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ../AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 4 --from-um-factor 20e-3 -s "LongRes4" -f "AuMieMediums/AllWaterTest/9)BoxDimensions/LongRes" -r 51.5 -pp "R" --wlen-range "np.array([250,800])" --index 1.33
mpirun --use-hwthread-cpus -np 6 python -m mpi4py ../AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 9 --from-um-factor 20e-3 -s "LongRes9" -f "AuMieMediums/AllWaterTest/9)BoxDimensions/LongRes" -r 51.5 -pp "R" --wlen-range "np.array([250,800])" --index 1.33
mpirun --use-hwthread-cpus -np 6 python -m mpi4py ../AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 13 --from-um-factor 20e-3 -s "LongRes13" -f "AuMieMediums/AllWaterTest/9)BoxDimensions/LongRes" -r 51.5 -pp "R" --wlen-range "np.array([250,800])" --index 1.33
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ../AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "RefLongRRes2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions" -r 51.5 -pp "R" --wlen-range "np.array([300,800])" --index 1.33
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ../AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "WLenMin250Res2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions/WLenMin/DoubleFreq" -r 51.5 -pp "R" --wlen-range "np.array([250,650])" --index 1.33
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ../AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "MoreAirFluxR0.0Res2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions/FluxR/MoreAir" -r 51.5 -pp "R" --wlen-range "np.array([500,650])" --index 1.33 --air-r-factor 2 --flux-r-factor 0.0
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ../AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "RefJCRes2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions" -r 51.5 -pp "JC" --wlen-range "np.array([500,650])" --index 1.33
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ../AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "RefLongJCRes2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions" -r 51.5 -pp "JC" --wlen-range "np.array([500,800])" --index 1.33
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ../AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "RefLongRRes2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions" -r 51.5 -pp "R" --wlen-range "np.array([500,800])" --index 1.33
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ../AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "RefRes2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions" -r 51.5 -pp "R" --wlen-range "np.array([500,650])" --index 1.33
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ../AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "FluxR0.30Res2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions/FluxR" -r 51.5 -pp "R" --wlen-range "np.array([500,650])" --index 1.33 --flux-r-factor 0.3
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ../AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "WLenMax700Res2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions/WLenMax" -r 51.5 -pp "R" --wlen-range "np.array([500,700])" --index 1.33
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ../AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "AirR06.5Res2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions/AirR" -r 51.5 -pp "R" --wlen-range "np.array([500,650])" --index 1.33 --air-r-factor 6.5
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ../AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "PMLWlen0.95Res2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions/PMLWlen" -r 51.5 -pp "R" --wlen-range "np.array([500,650])" --index 1.33 -pml 0.95
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ../AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "FluxR0.30Res2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions/FluxR" -r 51.5 -pp "R" --wlen-range "np.array([500,800])" --index 1.33 --flux-r-factor 0.1 --air-r-factor 1.0 -pml 0.5
