bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=50] span[ptile=1]" -W 0:59 -n 1 \
python eval.py 

