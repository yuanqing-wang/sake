for name in malonaldehyde azobenzene naphthalene paracetamol benzene_old ethanol toluene salicylic aspirin uracil
do
    bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=5] span[ptile=1]" -W 23:59 -n 1\
    python run.py $name
done
