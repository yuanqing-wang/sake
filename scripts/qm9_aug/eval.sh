for name in mu # alpha homo lumo gap r2 omega1 zpve_thermo U0_thermo U_thermo H_thermo G_thermo Cv_thermo

do
    bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=5] span[ptile=1]" -W 7:59 -n 1\
    python eval.py $name
done
