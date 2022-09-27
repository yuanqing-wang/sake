pip install "jax[tpu]==0.3.15" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax optax tensorflow tqdm
export PATH=$PATH:/home/wangy1/.local/bin
export PYTHONPATH=$PYTHONPATH:~/sake/
wget https://github.com/yuanqing-wang/sake/releases/download/v0.0.0/train.npz
wget https://github.com/yuanqing-wang/sake/releases/download/v0.0.0/valid.npz
wget https://github.com/yuanqing-wang/sake/releases/download/v0.0.0/test.npz
python3 run.py --target $target
