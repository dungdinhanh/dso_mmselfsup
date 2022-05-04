#PBS -l select=1:ncpus=20:ngpus=4
#PBS -l walltime=100:00:00
#PBS -N ssr18
#PBS -j oe
#PBS -o log/simsiamr18.log
#PBS -q gold
#PBS -v CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:20.06-py3

echo "CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:20.06-py3"


cd "$PBS_O_WORKDIR" || exit $?
echo ${PBS_O_WORKDIR}

USER=anhdung_dinh # Replace with your own HPC account name

nvidia-smi

export PYTHONPATH=/home/users/$USER/sutddev/mmselfsup/:$PYTHONPATH

cmd="bash tools/dist_train_cluster.sh ${USER} \
configs/selfsup/simsiam/simsiam_resnet18_4xb64-coslr-200e_in30p.py 4"
echo ${cmd}
eval ${cmd}
