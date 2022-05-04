#PBS -l select=1:ncpus=20:ngpus=4
#PBS -l walltime=100:00:00
#PBS -N densecl_linear
#PBS -j oe
#PBS -o log/moco_in30.log
#PBS -q project
#PBS -v CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:20.06-py3

echo "CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:20.06-py3"


cd "$PBS_O_WORKDIR" || exit $?
echo ${PBS_O_WORKDIR}

USER=dinh_anh_dung # Replace with your own HPC account name

nvidia-smi

export PYTHONPATH=/home/users/$USER/sutddev/mmselfsup/:$PYTHONPATH

cmd="bash tools/dist_train_cluster.sh ${USER} \
configs/selfsup/moco/mocov2_resnet50_4xb64-coslr-200e_in30p.py 4"
echo ${cmd}
eval ${cmd}
