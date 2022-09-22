#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=5]"
#BSUB -W 9:59
#BSUB -n 8
#BSUB -gpu "num=1/task:j_exclusive=yes:mode=shared"

unset LSB_AFFINITY_HOSTFILE
echo $LSB_AFFINITY_HOSTFILE

echo $LSB_HOSTS
hosts=$LSB_HOSTS


OMPI_MCA_opal_cuda_support=true MPI4JAX_USE_CUDA_MPI=1 mpirun \
        --hostfile $LSB_DJOB_HOSTFILE \
        -n 8 \
        -bind-to none -map-by slot \
        python run_mpi.py

