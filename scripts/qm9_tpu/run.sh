pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax optax tensorflow tqdm gdown
gdown 1KEh-DfNEWnmRlKuUFFQYT5rjYVZw6OeV
gdown 147gLT1x8LMNTfJX-EM6CLeWy4WYLVPuZ
gdown 1BhG2CZrZUz4C_M8TKiv9BqoTkbRi2kn0
export PYTHONPATH=$PYTHONPATH:~/sake/

learninte_rate = 1e-4
weight_decay = 1e-10
batch_size = 32

python3 run.py \
    --target $target \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --batch_size $batch_size

