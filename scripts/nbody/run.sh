for learning_rate in 1e-4; do
for weight_decay in 1e-10; do
for n_heads in 4; do

    bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=10] span[ptile=1]" -W 0:30 -n 1\
    python run.py \
        --learning_rate $learning_rate \
        --weight_decay $weight_decay \

done; done; done
