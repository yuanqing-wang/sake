for learning_rate in 1e-3; do
for weight_decay in 1e-8 1e-9 1e-10 1e-11; do
for n_heads in 4; do
for depth in 5; do
for hidden_features in 32; do

    bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=10] span[ptile=1]" -W 0:30 -n 1\
    python run.py \
        --learning_rate $learning_rate \
        --weight_decay $weight_decay \
        --n_heads $n_heads \
        --hidden_features $hidden_features \
        --depth $depth

done; done; done; done; done
