for name in U0 # alpha homo lumo gap r2 omega1 zpve U0 U H G Cv
do
    # bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=5] span[ptile=1]" -R V100 -W 119:59 -n 1\
    bsub -q cpuqueue -o %J.stdout -R "rusage[mem=1] span[ptile=30]" -W 8:59 -n 30 \
    python run.py $name
done
