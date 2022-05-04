#PBS -l select=1:ncpus=40:ngpus=8
#PBS -l walltime=100:00:00
#PBS -N ss
#PBS -j oe
#PBS -o log/simsiamr50.log
#PBS -q gold
#PBS -v CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:20.06-py3

echo "CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:20.06-py3"


cd "$PBS_O_WORKDIR" || exit $?
echo ${PBS_O_WORKDIR}

USER=anhdung_dinh # Replace with your own HPC account name

nvidia-smi

export PYTHONPATH=/home/users/$USER/sutddev/mmselfsup/:$PYTHONPATH

cmd="bash tools/dist_train_cluster.sh ${USER} \
configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k_cluster.py 8"
echo ${cmd}
eval ${cmd}
