#PBS -l select=1:ncpus=20:ngpus=4
#PBS -l walltime=100:00:00
#PBS -N ss_in30
#PBS -j oe
#PBS -o log/simsiam_in30.log
#PBS -q project
#PBS -v CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:20.06-py3

echo "CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:20.06-py3"


cd "$PBS_O_WORKDIR" || exit $?
echo ${PBS_O_WORKDIR}

USER=dinh_anh_dung # Replace with your own HPC account name

nvidia-smi

export PYTHONPATH=/home/users/$USER/sutddev/mmselfsup/:$PYTHONPATH

cmd="bash tools/dist_train_cluster.sh ${USER} \
configs/selfsup/simsiam_test/simsiam_resnet50_4xb64x8-coslr-200e_in30p_logmin.py 4"
echo ${cmd}
eval ${cmd}
