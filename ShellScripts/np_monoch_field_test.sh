### Test Periods
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -r 30 --wlen 405 --index 1 --time-period-factor 30 --folder "Field/NPMonoch/AuSphere/VacWatTest/TestPeriods/Vacuum" -s "VacNewPeriods405" -res 3
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -r 30 --wlen 405 --index 1.33 --time-period-factor 30 --folder "Field/NPMonoch/AuSphere/VacWatTest/TestPeriods/Water" -s "WatNewPeriods405" -res 3

### First Results WLen
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 405 --index 1 -f "Field/NPMonoch/AuSphere/VacWatField/TestWLen/Vacuum" -s "Norm405Res3" --time-period-factor 20
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 532 --index 1 -f "Field/NPMonoch/AuSphere/VacWatField/TestWLen/Vacuum" -s "Norm532Res3" --time-period-factor 20
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 642 --index 1 -f "Field/NPMonoch/AuSphere/VacWatField/TestWLen/Vacuum" -s "Norm642Res3" --time-period-factor 20
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 405 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatField/TestWLen/Water" -s "Norm405Res3" --time-period-factor 20
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 532 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatField/TestWLen/Water" -s "Norm532Res3" --time-period-factor 20
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 642 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatField/TestWLen/Water" -s "Norm642Res3" --time-period-factor 20

### Heavy Results WLen
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 5 --courant 0.25 -r 30 -wl 405 --index 1 -f "Field/NPMonoch/AuSphere/VacWatField/TestWLen/Vacuum" -s "Norm405Res5" --time-period-factor 20
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 5 --courant 0.25 -r 30 -wl 532 --index 1 -f "Field/NPMonoch/AuSphere/VacWatField/TestWLen/Vacuum" -s "Norm532Res5" --time-period-factor 20
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 5 --courant 0.25 -r 30 -wl 642 --index 1 -f "Field/NPMonoch/AuSphere/VacWatField/TestWLen/Vacuum" -s "Norm642Res5" --time-period-factor 20
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 5 --courant 0.25 -r 30 -wl 405 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatField/TestWLen/Water" -s "Norm405Res5" --time-period-factor 20
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 5 --courant 0.25 -r 30 -wl 532 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatField/TestWLen/Water" -s "Norm532Res5" --time-period-factor 20
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 5 --courant 0.25 -r 30 -wl 642 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatField/TestWLen/Water" -s "Norm642Res5" --time-period-factor 20

### Check empty-r-factor
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 532 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Water" -s "Empty532Res3ERF1.0" --time-period-factor 20 --empty-r-factor 1.0
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 532 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Water" -s "Empty532Res3ERF1.5" --time-period-factor 20 --empty-r-factor 1.5
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 532 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Water" -s "Empty532Res3ERF2.0" --time-period-factor 20 --empty-r-factor 2.0
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 532 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Water" -s "Empty532Res3ERF2.5" --time-period-factor 20 --empty-r-factor 2.5
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 532 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Water" -s "Empty532Res3ERF3.0" --time-period-factor 20 --empty-r-factor 3.0
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 532 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Water" -s "Empty532Res3ERF3.5" --time-period-factor 20 --empty-r-factor 3.5
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 532 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Water" -s "Empty532Res3ERF4.0" --time-period-factor 20 --empty-r-factor 4.5

# Check empty in all wlens
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 405 --index 1 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Vacuum	" -s "Empty405Res3ERF2.0" --time-period-factor 20 --empty-r-factor 2.0
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 532 --index 1 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Vacuum" -s "Empty532Res3ERF2.0" --time-period-factor 20 --empty-r-factor 2.0
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 642 --index 1 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Vacuum" -s "Empty642Res3ERF2.0" --time-period-factor 20 --empty-r-factor 2.0
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 405 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Water" -s "Empty405Res3ERF2.0" --time-period-factor 20 --empty-r-factor 2.0
#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 3 -r 30 -wl 642 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Water" -s "Empty642Res3ERF2.0" --time-period-factor 20 --empty-r-factor 2.0

#mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 5 --courant 0.25 -r 30 -wl 405 --index 1 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Vacuum" -s "Empty405Res5ERF2.0" --time-period-factor 20 --empty-r-factor 2.0
mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 5 --courant 0.25 -r 30 -wl 532 --index 1 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Vacuum" -s "Empty532Res5ERF2.0" --time-period-factor 20 --empty-r-factor 2.0
mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 5 --courant 0.25 -r 30 -wl 642 --index 1 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Vacuum" -s "Empty642Res5ERF2.0" --time-period-factor 20 --empty-r-factor 2.0
mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 5 --courant 0.25 -r 30 -wl 405 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Water" -s "Empty405Res5ERF2.0" --time-period-factor 20 --empty-r-factor 2.0
mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 5 --courant 0.25 -r 30 -wl 532 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Water" -s "Empty532Res5ERF2.0" --time-period-factor 20 --empty-r-factor 2.0
mpirun -np 4 python -m mpi4py AuMieSphere/u_np_monoch_field.py -res 5 --courant 0.25 -r 30 -wl 642 --index 1.33 -f "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Water" -s "Empty642Res5ERF2.0" --time-period-factor 20 --empty-r-factor 2.0
