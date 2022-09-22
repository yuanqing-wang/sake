#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -R "rusage[mem=50] span[ptile=1]"
#BSUB -W 0:10
#BSUB -n 1

python init.py

