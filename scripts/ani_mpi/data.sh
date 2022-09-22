#BSUB -o %J.stdout
#BSUB -R "rusage[mem=100] span[ptile=1]"
#BSUB -W 0:59
#BSUB -n 1

python data.py 

