#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -R "rusage[mem=50] span[ptile=1]"
#BSUB -W 1:00
#BSUB -n 1

python eval.py

