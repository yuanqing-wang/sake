#BSUB -R V100
#BSUB -W 1:00
#BSUB -R "rusage[mem=10] span[ptile=1]"
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -q gpuqueue

python run.py

