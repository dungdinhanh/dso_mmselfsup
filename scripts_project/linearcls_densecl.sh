#PBS -l select=1:ncpus=20:ngpus=2
#PBS -l walltime=72:00:00
#PBS -N densecl_linear
#PBS -j oe
#PBS -o log/densecl_linearcls.log
#PBS -q project
#PBS -v CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:20.06-py3

echo "CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:20.06-py3"


cd "$PBS_O_WORKDIR" || exit $?
echo ${PBS_O_WORKDIR}

USER=dinh_anh_dung # Replace with your own HPC account name

nvidia-smi

export PYTHONPATH=/home/users/$USER/sutddev/mmselfsup/:$PYTHONPATH

cmd="bash tools/benchmarks/classification/dist_train_linear_cluster.sh ${USER} \
 configs/benchmarks/classification/imagenet/resnet50_4xb64-coslr-100e_in1k.py \
 ../../scratch/dso/openss/work_dirs/downloads/densecl_resnet50_8xb32-coslr-200e_in1k_20211214-1efb342c.pth 2"
 echo ${cmd}
 eval ${cmd}
