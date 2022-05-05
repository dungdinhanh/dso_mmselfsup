<div align="center">
  <img src="./resources/tl.png" width="500"/>

[![PyPI](https://img.shields.io/pypi/v/mmselfsup)]()
[![docs](https://img.shields.io/badge/docs-latest-blue)]()
[![badge](https://github.com/open-mmlab/mmselfsup/workflows/build/badge.svg)]()
[![codecov](https://codecov.io/gh/open-mmlab/mmselfsup/branch/master/graph/badge.svg)]()
[![license](https://img.shields.io/github/license/open-mmlab/mmselfsup.svg)]()

</div>

## SimDis for SimSiam

This is the modified version of the SimDis for [SimSiam](algorithms/ss.md) instead of BYOL as in the [original paper](https://arxiv.org/pdf/2106.11304.pdf)



The main idea of the SimDis work is to match the views of students to teachers.
The loss is provided as: <img src="https://render.githubusercontent.com/render/math?math=L = \frac{1}{2} D(p^T_1, p^S_1)+ \frac{1}{2} D(p^T_2, p^S_2)">
In order to provid 