#BSUB -R 2080
#BSUB -W 0:10
#BSUB -R "rusage[mem=50] span[ptile=1]"
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -q gpuqueue

python speed.py

