export PATH=/nfs/home/vpais/miniconda:$PATH
conda activate pmp
cd /nfs/home/vpais/ThesisPython
python -m mpi4py ./AuMieSphere/u_au_mie_scattering.py -np 64 --parallel True
