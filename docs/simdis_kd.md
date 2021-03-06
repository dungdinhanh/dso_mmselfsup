<div align="center">
  <img src="../resources/tl.png" width="500"/>

[![PyPI](https://img.shields.io/pypi/v/mmselfsup)]()
[![docs](https://img.shields.io/badge/docs-latest-blue)]()
[![badge](https://github.com/open-mmlab/mmselfsup/workflows/build/badge.svg)]()
[![codecov](https://codecov.io/gh/open-mmlab/mmselfsup/branch/master/graph/badge.svg)]()
[![license](https://img.shields.io/github/license/open-mmlab/mmselfsup.svg)]()

</div>

## SimDis for SimSiam

This is the modified version of the SimDis for [SimSiam](algorithms/ss.md) instead of BYOL as in the [original paper](https://arxiv.org/pdf/2106.11304.pdf)



The main idea of the SimDis work is to match the views of students to teachers.
The loss is provided as: <img src="https://render.githubusercontent.com/render/math?math=L = \frac{1}{2} D(p^T_1, p^S_1) %2b  \frac{1}{2} D(p^T_2, p^S_2)">
Where <img src="https://render.githubusercontent.com/render/math?math=D, p^T_1, p^T_2, p^S_1, p^S_2"> are cosine similarity, first view, second view prediction of teacher and student respectively.

Fig.1 shows the paradigm of the SimDisSiam where the predictions of students are matched with the predictions of teacher
correspondingly.

<div align="center">
<figure>
    <img src="../resources/DSO/report3/view1.png" style="width:60%"\><br>  
     <figcaption align = "center">Fig.1 <b>SimDisSiam model</b>: Matching between views of students
     and teacher</figcaption>
</figure>
</div>

In order to run the SimDisSiam code, there are several steps

**First step** is to train the SimDisSiam model by running the following code:

``bash tools/dist_train_kd.sh configs/selfsup/simdis/simdis_resnet18_4xb64-coslr-200e_in30p.py 4``

The above code is set for training self-supervised SimDis for Resnet on ImageNet30% with 4 GPUs. To make changes to the number of GPUs,
please refer the [README.md](../README.md) with original guidances from OpenMMselfsup to change the configs.

**Second step** is to extract the backbone of the self-supervised model via command;

``python tools/model_converters/extract_backbone_weight.py work_dirs/selfsup/simdis/simdis_resnet18_4xb64-coslr-200e_in30p/latest.pth
work_dirs/selfsup/simdis/simdis_resnet18_4xb64-coslr-200e_in30p/backbone_simdissiam.pth``

(Remember to change the path if the config is changed)

**Third step** is to evaluate the performance of the self-supervised mode via command:

``bash tools/benchmark/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_2xb128-steplr-100e_in1k.py
 work_dirs/selfsup/simdis/simdis_resnet18_4xb64-coslr-200e_in30p/backbone_simdissiam.pth``
 

