for name in U_thermo
do
    bsub -q gpuqueue -R V100 -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=5] span[ptile=1]" -W 23:59 -n 1\
    python run.py $name
done
