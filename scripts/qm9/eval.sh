for name in U0 U H G Cv mu alpha homo lumo gap r2 omega1 zpve
do
    bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=10] span[ptile=1]" -W 7:59 -n 1\
    python eval.py $name
done
