#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=20] span[ptile=1]"
#BSUB -W 23:59
#BSUB -n 1
#BSUB -gpu "num=1/task:j_exclusive=yes:mode=shared"

source ~/.bashrc
conda activate mpi
module load cuda/11.3

python run.py

