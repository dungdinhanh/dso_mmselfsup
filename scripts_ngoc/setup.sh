#PBS -l select=1:ncpus=2:ngpus=1
#PBS -N setup
#PBS -j oe
#PBS -o setup.log
#PBS -q gold
#PBS -v CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:20.06-py3

echo "CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:20.06-py3"


cd "$PBS_O_WORKDIR" || exit $?
echo ${PBS_O_WORKDIR}

USER=thibaongoc_nguyen # Replace with your own HPC account name


cmd="/opt/conda/bin/conda create --force -y -n openss python==3.8"
echo ${cmd}
#eval ${cmd}
/home/users/$USER/.conda/envs/openss/bin/pip config set global.target /home/users/$USER/.conda/envs/openss/lib/python3.8/site-packages/
# source /home/users/$USER/.bashrc
# export PATH=/home/users/$USER/.conda/envs/openss/bin/:$PATH
# export PYTHONPATH=/home/users/$USER/.conda/envs/openss/lib/python3.8/site-packages/:$PYTHONPATH
# export PYTHONPATH=/home/users/$USER/sutddev/mmselfsup/:$PYTHONPATH



cmd="/home/users/$USER/.conda/envs/openss/bin/pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0"
echo ${cmd}
#eval ${cmd}

cmd="/home/users/$USER/.conda/envs/openss/bin/pip install  -v -e ."
echo ${cmd}
#eval ${cmd}

cmd="/home/users/$USER/.conda/envs/openss/bin/pip install opencv-python-headless"
echo ${cmd}
#eval ${cmd}

cmd="/home/users/$USER/.conda/envs/openss/bin/pip install  mmcv==1.4.0"
echo ${cmd}
#eval ${cmd}


cmd="/home/users/$USER/.conda/envs/openss/bin/pip install  mmsegmentation mmdet"
echo ${cmd}
#eval ${cmd}

cmd="/home/users/$USER/.conda/envs/openss/bin/pip install pandas"
echo ${cmd}
#eval ${cmd}

cmd="/home/users/$USER/.conda/envs/openss/bin/pip install lpips"
echo ${cmd}
eval ${cmd}