# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import time
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck, build_algorithm
from mmcv.runner.checkpoint import load_checkpoint
from .base import BaseModel
import torch
import torch.nn.functional as F
from mmselfsup.models.algorithms.simsiam_kd import *
from collections import OrderedDict
import torch.distributed as dist

@ALGORITHMS.register_module()
class SimDis_poskd_sim_pred_target(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimDis_poskd_sim_pred_target, self).__init__(
            backbone,
            neck=neck,
            head=head,
            init_cfg=init_cfg,
            **kwargs
        )


    def forward_train(self, img):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.
        Returns:
            loss[str, Tensor]: A dictionary of loss components
        """
        assert isinstance(img, list)
        self.teacher.eval()
        img_v1 = img[0]
        img_v2 = img[1]
        img_v3 = img[2]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]
        zt3 = self.teacher.encoder(img_v3)[0]

        p1 = self.head(z1, z2, loss_cal=False)
        p2 = self.head(z2, z1, loss_cal=False)

        pt1 = self.teacher.head(zt1, zt2, loss_cal=False)
        pt2 = self.teacher.head(zt2, zt1, loss_cal=False)
        pt3 = self.teacher.head(zt3, zt1, loss_cal=False)

        student_cs1 = cosine_sim(p1, z2, False)
        student_cs2 = cosine_sim(p2, z1, False)

        teacher_cs1 = cosine_sim(pt1, zt2, False).detach()
        teacher_cs2 = cosine_sim(pt2, zt1, False).detach()


        loss_pos_kd = 0.5 * (nn.functional.mse_loss(student_cs1, teacher_cs1) +
                             nn.functional.mse_loss(student_cs2, teacher_cs2))

        # simsiam_loss = 0.5 * (cosine_sim(p1, z2) + cosine_sim(p2, z1))

        distillation_loss1 = cosine_sim(p1, pt1, False)
        distillation_loss2 = cosine_sim(p2, pt2, False)

        distillation_target1 = cosine_sim(pt1, pt3, False).detach()
        distillation_target2 = cosine_sim(pt2, pt3, False).detach()

        # distillation_loss = 0.5 * (distillation_loss1 + distillation_loss2)

        distillation_loss = 0.5 * (nn.functional.mse_loss(distillation_loss1, distillation_target1) + \
                                   nn.functional.mse_loss(distillation_loss2, distillation_target2))

        losses = loss_pos_kd + distillation_loss

        return dict(loss=losses)


@ALGORITHMS.register_module()
class SimDis_nogt_sim_pred_target(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimDis_nogt_sim_pred_target, self).__init__(
            backbone,
            neck=neck,
            head=head,
            init_cfg=init_cfg,
            **kwargs
        )


    def forward_train(self, img):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.
        Returns:
            loss[str, Tensor]: A dictionary of loss components
        """
        assert isinstance(img, list)
        self.teacher.eval()
        img_v1 = img[0]
        img_v2 = img[1]
        img_v3 = img[2]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]
        zt3 = self.teacher.encoder(img_v3)[0]

        p1 = self.head(z1, z2, loss_cal=False)
        p2 = self.head(z2, z1, loss_cal=False)

        pt1 = self.teacher.head(zt1, zt2, loss_cal=False)
        pt2 = self.teacher.head(zt2, zt1, loss_cal=False)
        pt3 = self.teacher.head(zt3, zt1, loss_cal=False)




        loss_pos_kd = 0.0

        # simsiam_loss = 0.5 * (cosine_sim(p1, z2) + cosine_sim(p2, z1))

        distillation_loss1 = cosine_sim(p1, pt1, False)
        distillation_loss2 = cosine_sim(p2, pt2, False)

        distillation_target1 = cosine_sim(pt1, pt3, False).detach()
        distillation_target2 = cosine_sim(pt2, pt3, False).detach()

        # distillation_loss = 0.5 * (distillation_loss1 + distillation_loss2)

        distillation_loss = 0.5 * (nn.functional.mse_loss(distillation_loss1, distillation_target1) + \
                                   nn.functional.mse_loss(distillation_loss2, distillation_target2))

        losses = loss_pos_kd + distillation_loss

        return dict(loss=losses)