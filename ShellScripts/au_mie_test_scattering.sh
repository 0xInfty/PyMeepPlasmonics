#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "FluxR0.30Res2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions/FluxR" -r 51.5 -pp "R" --wlen-range "np.array([500,650])" --index 1.33 --flux-r-factor 0.3
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "AirR06.5Res2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions/AirR" -r 51.5 -pp "R" --wlen-range "np.array([500,650])" --index 1.33 --air-r-factor 6.5
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "PMLWlen0.95Res2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions/PMLWlen" -r 51.5 -pp "R" --wlen-range "np.array([500,650])" --index 1.33 -pml 0.95
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "WLenMax700Res2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions/WLenMax" -r 51.5 -pp "R" --wlen-range "np.array([500,700])" --index 1.33
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "WLenMin500Res2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions/WLenMin" -r 51.5 -pp "R" --wlen-range "np.array([500,650])" --index 1.33
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "TFCell0.2Res2" -f "AuMieSphere/AuMie/13)TestPaper/4)PaperJCFit/TestTFCell/AllVac450650" -r 51.5 -pp "JC" --wlen-range "np.array([450,650])" --time-factor-cell 0.2
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "STFactor01.0Res2" -f "AuMieSphere/AuMie/13)TestPaper/4)PaperJCFit/TestSTFactor/AllVac450650" -r 51.5 -pp "JC" --wlen-range "np.array([450,650])" --second-time-factor 1
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "Courant0.50Res2" -f "AuMieMediums/AllWaterTest/6)Courant/500to700" -r 51.5 -pp "R" --wlen-range "np.array([500,700])" --courant .5 --index 1.33
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -res 2 --from-um-factor 10e-3 -s "RefRRes2" -f "AuMieMediums/AllWaterTest/9)BoxDimensions/RefIdeal/3)NewRef" -r 51.5 -pp "R" --wlen-range "np.array([500,650])" --index 1.33
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -f "Test/TestChunks" -s "AllWatChunksEvenlyFalseNP6" -res 2 --split-chunks-evenly False --index 1.33 --wlen-range "[500, 650]"
############
#python ./AuMieSphere/u_au_mie_scattering.py --parallel False -f "Test/TestRAM" -s "TestRAMConsole" -res 2 --wlen-range "[450, 600]" --load-flux False
#mpirun --use-hwthread-cpus -np 1 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 1 --parallel True -f "Test/TestRAM" -s "TestRAMParallel1" -res 2 --load-flux False
#mpirun --use-hwthread-cpus -np 2 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 2 --parallel True -f "Test/TestRAM" -s "TestRAMParallel2" -res 2 --load-flux False
#mpirun --use-hwthread-cpus -np 3 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 3 --parallel True -f "Test/TestRAM" -s "TestRAMParallel3" -res 2 --load-flux False
#mpirun --use-hwthread-cpus -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 4 --parallel True -f "Test/TestRAM" -s "TestRAMParallel4" -res 2 --load-flux False
#mpirun --use-hwthread-cpus -np 5 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 5 --parallel True -f "Test/TestRAM" -s "TestRAMParallel5" -res 2 --load-flux False
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -f "Test/TestRAM" -s "TestRAMParallel6" -res 2 --load-flux False
#mpirun --use-hwthread-cpus -np 7 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 7 --parallel True -f "Test/TestRAM" -s "TestRAMParallel7" -res 2 --load-flux False
#mpirun --use-hwthread-cpus -np 8 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 8 --parallel True -f "Test/TestRAM" -s "TestRAMParallel8" -res 2 --load-flux False
#############
#mpirun --use-hwthread-cpus -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 4 --parallel True -f "Test/TestRAM" -s "AllVacRes2" -res 2 --wlen-range "[450, 600]" --load-flux False
#mpirun --use-hwthread-cpus -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 4 --parallel True -f "Test/TestRAM" -s "AllWatRes2" -res 2 --wlen-range "[500, 650]" --index 1.33 --load-flux False
#mpirun --use-hwthread-cpus -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 4 --parallel True -f "Test/TestRAM" -s "AllVacRes3" -res 3 --wlen-range "[450, 600]" --load-flux False
#mpirun --use-hwthread-cpus -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 4 --parallel True -f "Test/TestRAM" -s "AllWatRes3" -res 3 --wlen-range "[500, 650]" --index 1.33 --load-flux False
#mpirun --use-hwthread-cpus -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 4 --parallel True -f "Test/TestRAM" -s "AllVacRes4" -res 4 --wlen-range "[450, 600]" --load-flux False
#mpirun --use-hwthread-cpus -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 4 --parallel True -f "Test/TestRAM" -s "AllWatRes4" -res 4 --wlen-range "[500, 650]" --index 1.33 --load-flux False
############
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -f "Test/TestRAM" -s "AllVacNear2FarTrueRes2" -res 2 --near2far True --wlen-range "[450, 600]" --load-flux False
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -f "Test/TestRAM" -s "AllVacNear2FarFalseRes2" -res 2 --near2far False --wlen-range "[450, 600]" --load-flux False
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -f "Test/TestRAM" -s "AllWatNear2FarTrueRes2" -res 2 --near2far True --wlen-range "[500, 650]" --index 1.33 --load-flux False --flux-r-factor 0.3
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -f "Test/TestRAM" -s "AllWatNear2FarFalseRes2" -res 2 --near2far False --wlen-range "[500, 650]" --index 1.33 --load-flux False --flux-r-factor 0.3
##############
#mpirun --use-hwthread-cpus -np 6 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 6 --parallel True -f "Test/TestRAM" -s "TestSWAP" -res 4
#mpirun --use-hwthread-cpus -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 4 --parallel True -f "Test/TestRAM" -s "TestSWAP" -res 4
##############
#mpirun --use-hwthread-cpus -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 4 --parallel True -f "Test/TestNP4" -s "TestHWThreadsTrue" -res 2 --load-flux False
#mpirun -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 4 --parallel True -f "Test/TestNP4" -s "TestHWThreadsFalse" -res 2 --load-flux False
#############
#mpirun -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 4 --material "Ag" --wlen-range "[350, 500]" -res 4 -s "SilverRes4" -f "Test/TestSilver"
#mpirun -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 4 --material "Ag" --wlen-range 350 500 -r 30 -res 2 -s "SilverDiam60Res2" -f "Test/TestSilver"
#mpirun -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 4 --material "Ag" --paper "JC" --wlen-range 350 500 -res 5 -s "SilverJCRes5" -f "Test/TestSilver"
#############
mpirun -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -nc 4 -res 4 -s "Over00" -f "Test/TestGlass/OverlapRes4" --wlen-range "[500, 700]" --index 1.33 --surface-index 1.54
#mpirun -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -nc 4 -res 4 -s "Over20" -f "Test/TestGlass/OverlapRes4" --wlen-range "[500, 700]" --index 1.33 --surface-index 1.54 --overlap 20
mpirun -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -nc 4 -res 4 -s "Over10" -f "Test/TestGlass/OverlapRes4" --wlen-range "[500, 700]" --index 1.33 --surface-index 1.54 --overlap 10
#mpirun -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -nc 4 -res 4 -s "Over25" -f "Test/TestGlass/OverlapRes4" --wlen-range "[500, 700]" --index 1.33 --surface-index 1.54 --overlap 25
mpirun -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -nc 4 -res 4 -s "Over15" -f "Test/TestGlass/OverlapRes4" --wlen-range "[500, 700]" --index 1.33 --surface-index 1.54 --overlap 15
#mpirun -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -nc 4 -res 4 -s "Over05" -f "Test/TestGlass/OverlapRes4" --wlen-range "[500, 700]" --index 1.33 --surface-index 1.54 --overlap 5
############ DO I ADD OVERLAP IN THIS RESOLUTION RUN?
#mpirun -np 4 python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -nc 4 --from-um-factor 20e-3 -res 8 -s "GlassRes8" -f "Test/TestGlass/TestGlassRes" --wlen-range "[500, 700]" --index 1.33 --surface-index 1.54
