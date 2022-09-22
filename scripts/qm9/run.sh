for name in mu alpha homo lumo gap r2 omega1 zpve U0 U H G Cv
do
    bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=5] span[ptile=1]" -W 23:59 -n 1\
    python run.py $name
done
