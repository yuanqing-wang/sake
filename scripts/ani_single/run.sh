#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=4:j_exclusive=yes"
#BSUB -R "rusage[mem=50] span[ptile=1]"
#BSUB -W 23:59
#BSUB -n 1

python run_gpu.py

