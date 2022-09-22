for name in benzene_old malonaldehyde azobenzene naphthalene paracetamol benzene_old ethanol toluene salicylic aspirin uracil
do
    bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=5] span[ptile=1]" -W 7:59 -n 1\
    python eval.py $name
done
