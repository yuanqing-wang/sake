#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=50] span[ptile=1]"
#BSUB -W 23:59
#BSUB -R V100
#BSUB -n 1

python run.py 

