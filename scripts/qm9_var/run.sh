for name in U # mu alpha homo lumo gap r2 omega1 zpve U H G Cv # U0
do
    bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=10] span[ptile=1]" -W 47:59 -n 1\
    python run.py $name
done
