for name in mu alpha homo lumo gap r2 omega1 zpve U H G Cv U0; do
for learning_rate in 1e-5; do
for weight_decay in 1e-10; do
for batch_size in 16; do

    bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R V100 -R "rusage[mem=10] span[ptile=1]" -W 47:59 -n 1\
    python run.py \
        --target $name \
        --learning_rate $learning_rate \
        --weight_decay $weight_decay \
        --batch_size $batch_size

done; done; done; done
