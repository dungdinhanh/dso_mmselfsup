<div align="center">
  <img src="./resources/tl.png" width="500"/>

[![PyPI](https://img.shields.io/pypi/v/mmselfsup)]()
[![docs](https://img.shields.io/badge/docs-latest-blue)]()
[![badge](https://github.com/open-mmlab/mmselfsup/workflows/build/badge.svg)]()
[![codecov](https://codecov.io/gh/open-mmlab/mmselfsup/branch/master/graph/badge.svg)]()
[![license](https://img.shields.io/github/license/open-mmlab/mmselfsup.svg)]()

</div>

## Introduction

The work is to provide a knowledge distillation scheme for self-supervised learning. This aims to solve one fact that the self-supervised scheme works very well for large networks such as Resnet 50, yet the performance is unexepectedly lower in small networks such as Mobilenet-v2 and Resnet18


The master branch works with **PyTorch 1.5** or higher.


### Major features

- **Methods All in One**

  Similar to MMSelfsup. The code provides a wide range of state-of-the-art methods in self-supervised learning. 
  
  For comprehensive comparison in all benchmarks, most of the pre-training methods are under the same setting.

- **Standardized Benchmarks**

  DSO MMSelfSup standardizes the benchmarks including logistic regression, SVM / Low-shot SVM from linearly probed features, semi-supervised classification, object detection and semantic segmentation.

- **Compatibility**

  Since MMSelfSup adopts similar design of modulars and interfaces as those in other OpenMMLab projects, this project
   supports smooth evaluation on downstream tasks with other OpenMMLab projects like object detection and segmentation.


## License

This project is released under the [Apache 2.0 license](LICENSE).


## Model Zoo and Benchmark

### Model Zoo
Please refer to [model_zoo.md](docs/model_zoo.md) for a comprehensive set of pre-trained models and benchmarks.

Supported algorithms from MMSelfsup: 

- [x] [Relative Location (ICCV'2015)](https://arxiv.org/abs/1505.05192)
- [x] [Rotation Prediction (ICLR'2018)](https://arxiv.org/abs/1803.07728)
- [x] [DeepCLuster (ECCV'2018)](https://arxiv.org/abs/1807.05520)
- [x] [NPID (CVPR'2018)](https://arxiv.org/abs/1805.01978)
- [x] [ODC (CVPR'2020)](https://arxiv.org/abs/2006.10645)
- [x] [MoCo v1 (CVPR'2020)](https://arxiv.org/abs/1911.05722)
- [x] [SimCLR (ICML'2020)](https://arxiv.org/abs/2002.05709)
- [x] [MoCo v2 (ArXiv'2020)](https://arxiv.org/abs/2003.04297)
- [x] [BYOL (NeurIPS'2020)](https://arxiv.org/abs/2006.07733)
- [x] [SwAV (NeurIPS'2020)](https://arxiv.org/abs/2006.09882)
- [x] [DenseCL (CVPR'2021)](https://arxiv.org/abs/2011.09157)
- [x] [SimSiam (CVPR'2021)](https://arxiv.org/abs/2011.10566)

Other algorithms developed by us:

- [x] [SimDis for SimSiam (arxiv)](docs/simdis_kd.md)
- [x] [NegKD](docs/neg_kd.md)
- [x] [PosKD](docs/pos_kd.md)
- [x] [PosNegKD]()
- [x] [SimDisPosKD]()
- [x] [SimDisNegKD]()

Please refers to this [doc](docs/models_kd.md) to see more. 

### Benchmark

  | Benchmarks                                   | Setting                                                                                                                                                              |
  | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | ImageNet Linear Classification (Multi-head)  | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
  | ImageNet Linear Classification (Last)        |                                                                                                                                                                      |
  | ImageNet Semi-Sup Classification             |                                                                                                                                                                      |
  | Places205 Linear Classification (Multi-head) | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
  | iNaturalist2018 Linear Classification (Multi-head) | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf)               |
  | PASCAL VOC07 SVM                             | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
  | PASCAL VOC07 Low-shot SVM                    | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
  | PASCAL VOC07+12 Object Detection             | [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)               |
  | COCO17 Object Detection                      | [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)               |
  | Cityscapes Segmentation                      | [MMSeg](configs/benchmarks/mmsegmentation/cityscapes/fcn_r50-d8_769x769_40k_cityscapes.py)                                                                           |
  | PASCAL VOC12 Aug Segmentation                | [MMSeg](configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py)                                                                               |

## Installation

run `bash scripts/setup.sh` for installation. Remember to create a virtual env first before installing (venv or virtualenv)

## Data preparation 

[prepare_data.md](docs/prepare_data.md) for dataset preparation.

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of the project.

Refer to MMselfsup docs for more guidances:
- [config](docs/tutorials/0_config.md)
- [add new dataset](docs/tutorials/1_new_dataset.md)
- [data pipeline](docs/tutorials/2_data_pipeline.md)
- [add new module](docs/tutorials/3_new_module.md)
- [customize schedules](docs/tutorials/4_schedule.md)
- [customize runtime](docs/tutorials/5_runtime.md)
- [benchmarks](docs/tutorials/6_benchmarks.md)

## Acknowledgement

Remarks:

The code is developed from the [MMselfsup](https://github.com/open-mmlab/mmselfsup). 

MMSelfSup is an open source self-supervised representation learning toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

