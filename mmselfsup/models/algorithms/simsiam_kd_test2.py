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
import lpips
from mmselfsup.models.algorithms.simsiam_kd_test import *


@ALGORITHMS.register_module()
class SimSiam_dimcollapsecheckLPIPS2(SimSiamKD_OLMH_dimcollapsecheck):  # optimize lower, minimize Higher
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiam_dimcollapsecheckLPIPS2, self).__init__(backbone, neck, head, init_cfg, **kwargs)
        self.fn_alex = lpips.LPIPS(net='alex')

    def forward_train(self, img, org_img):
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
        org_img = org_img[0]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        o_head_student1 = self.head(z1, z2)
        o_head_student2 = self.head(z2, z1)

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        o_head_teacher1 = self.teacher.head(zt1, zt2)
        o_head_teacher2 = self.teacher.head(zt2, zt1)

        teacher_loss1 = o_head_teacher1['cossim'].detach()
        teacher_loss2 = o_head_teacher2['cossim'].detach()

        student_loss1 = o_head_student1['cossim']
        student_loss2 = o_head_student2['cossim']

        # for plotting loss
        l_s1 = torch.mean(student_loss1)
        l_s2 = torch.mean(student_loss2)
        l_s = 1 / 2 * (l_s1 + l_s2).detach()
        l_t = 1 / 2 * (torch.mean(teacher_loss1) + torch.mean(teacher_loss2)).detach()

        index_lower1 = student_loss1 < teacher_loss1
        index_lower2 = student_loss2 < teacher_loss2

        index_higher1 = torch.bitwise_not(index_lower1)
        index_higher2 = torch.bitwise_not(index_lower2)

        # log frac
        num1 = float(torch.nonzero(index_lower1).size(0))
        num2 = float(torch.nonzero(index_lower2).size(0))
        total_number = float(img_v1.size(0))
        frac = (num1 + num2) / (total_number * 2)
        # put test here
        # log lpips
        d = self.fn_alex(img_v1, img_v2)

        # predictions
        p1 = o_head_student1['pred']
        p2 = o_head_student2['pred']

        pt1 = o_head_teacher1['pred']
        pt2 = o_head_teacher2['pred']


        # log image
        if self.save_images:
            images_lower1 = img[0][index_lower1]
            images_lower1_proj = img[1][index_lower1]
            images_higher1 = img[0][index_higher1]
            images_higher2_proj = img[1][index_higher1]
            ori_lower1 = org_img[index_lower1]
            ori_higher1 = org_img[index_higher1]

            set_1 = [images_lower1, images_lower1_proj, ori_lower1, student_loss1[index_lower1],
                     teacher_loss1[index_lower1], d[index_lower1],
                     images_higher1, images_higher2_proj, ori_higher1, student_loss1[index_higher1],
                     teacher_loss1[index_higher1], d[index_higher1],
                     p1[index_lower1], p2[index_lower1], pt1[index_lower1], pt2[index_lower1],
                     p1[index_higher1], p2[index_higher1], pt1[index_higher1], pt2[index_higher1]]

            # images_lower2 = img[0][index_lower2]
            # images_lower2_proj = img[1][index_lower2]
            # images_higher2 = img[0][index_higher2]
            # images_higher2_proj = img[1][index_higher2]
            # ori_lower2 = org_img[index_lower2]
            # ori_higher2 = org_img[index_higher2]
            #
            # set_2 = [images_lower2, images_lower2_proj, ori_lower2, student_loss2[index_lower2],
            #          teacher_loss2[index_lower2], d[index_lower2],
            #          images_higher2, images_higher2_proj, ori_higher2, student_loss2[index_higher2],
            #          teacher_loss2[index_higher2], d[index_higher2]]
            set_2 = None
        else:
            set_1, set_2 = None, None

        loss1 = torch.mean(student_loss1)
        loss2 = torch.mean(student_loss2)

        losses = 1 / 2 * (loss1 + loss2)
        return dict(loss=losses, l_student=l_s, l_teacher=l_t, frac=torch.tensor(frac).cuda()), set_1, set_2


    def val_step(self, data, optimizer, teacher_model, save_images):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """

        if self.teacher is None:
            self.teacher = teacher_model

        self.save_images = save_images[0]
        losses, set_1, set_2 = self(**data)
        loss, log_vars = self._parse_losses(losses)

        if isinstance(data['img'], list):
            num_samples = len(data['img'][0].data)
        else:
            num_samples = len(data['img'].data)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs, set_1, set_2


@ALGORITHMS.register_module()
class SimSiam_dimcollapsecheckLPIPS_fixedfirstview(SimSiam_dimcollapsecheckLPIPS2):  # optimize lower, minimize Higher
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiam_dimcollapsecheckLPIPS_fixedfirstview, self).__init__(backbone, neck, head, init_cfg, **kwargs)
        self.img_v1 = None

    def forward_train(self, img, org_img):
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
        if self.img_v1 == None:
            self.img_v1 = img[0]
        img_v1 = self.img_v1
        img_v2 = img[1]
        org_img = org_img[0]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        o_head_student1 = self.head(z1, z2)
        o_head_student2 = self.head(z2, z1)

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        o_head_teacher1 = self.teacher.head(zt1, zt2)
        o_head_teacher2 = self.teacher.head(zt2, zt1)

        teacher_loss1 = o_head_teacher1['cossim'].detach()
        teacher_loss2 = o_head_teacher2['cossim'].detach()

        student_loss1 = o_head_student1['cossim']
        student_loss2 = o_head_student2['cossim']

        # for plotting loss
        l_s1 = torch.mean(student_loss1)
        l_s2 = torch.mean(student_loss2)
        l_s = 1 / 2 * (l_s1 + l_s2).detach()
        l_t = 1 / 2 * (torch.mean(teacher_loss1) + torch.mean(teacher_loss2)).detach()

        index_lower1 = student_loss1 < teacher_loss1
        index_lower2 = student_loss2 < teacher_loss2

        index_higher1 = torch.bitwise_not(index_lower1)
        index_higher2 = torch.bitwise_not(index_lower2)

        # log frac
        num1 = float(torch.nonzero(index_lower1).size(0))
        num2 = float(torch.nonzero(index_lower2).size(0))
        total_number = float(img_v1.size(0))
        frac = (num1 + num2) / (total_number * 2)
        # put test here
        # log lpips
        d = self.fn_alex(img_v1, img_v2)

        # predictions
        p1 = o_head_student1['pred']
        p2 = o_head_student2['pred']

        pt1 = o_head_teacher1['pred']
        pt2 = o_head_teacher2['pred']


        # log image
        if self.save_images:
            images_lower1 = img_v1[index_lower1]
            images_lower1_proj = img_v2[index_lower1]
            images_higher1 = img_v1[index_higher1]
            images_higher2_proj = img_v2[index_higher1]
            ori_lower1 = org_img[index_lower1]
            ori_higher1 = org_img[index_higher1]

            set_1 = [images_lower1, images_lower1_proj, ori_lower1, student_loss1[index_lower1],
                     teacher_loss1[index_lower1], d[index_lower1],
                     images_higher1, images_higher2_proj, ori_higher1, student_loss1[index_higher1],
                     teacher_loss1[index_higher1], d[index_higher1],
                     p1[index_lower1], p2[index_lower1], pt1[index_lower1], pt2[index_lower1],
                     p1[index_higher1], p2[index_higher1], pt1[index_higher1], pt2[index_higher1]]

            # images_lower2 = img[0][index_lower2]
            # images_lower2_proj = img[1][index_lower2]
            # images_higher2 = img[0][index_higher2]
            # images_higher2_proj = img[1][index_higher2]
            # ori_lower2 = org_img[index_lower2]
            # ori_higher2 = org_img[index_higher2]
            #
            # set_2 = [images_lower2, images_lower2_proj, ori_lower2, student_loss2[index_lower2],
            #          teacher_loss2[index_lower2], d[index_lower2],
            #          images_higher2, images_higher2_proj, ori_higher2, student_loss2[index_higher2],
            #          teacher_loss2[index_higher2], d[index_higher2]]
            set_2 = None
        else:
            set_1, set_2 = None, None

        loss1 = torch.mean(student_loss1)
        loss2 = torch.mean(student_loss2)

        losses = 1 / 2 * (loss1 + loss2)
        return dict(loss=losses, l_student=l_s, l_teacher=l_t, frac=torch.tensor(frac).cuda()), set_1, set_2


@ALGORITHMS.register_module()
class SimSiam_dimcollapsecheckLPIPS_fixedsecondview(SimSiam_dimcollapsecheckLPIPS2):  # optimize lower, minimize Higher
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiam_dimcollapsecheckLPIPS_fixedsecondview, self).__init__(backbone, neck, head, init_cfg, **kwargs)
        self.img_v2 = None

    def forward_train(self, img, org_img):
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
        if self.img_v2 == None:
            self.img_v2 = img[1]
        img_v1 = img[0]
        img_v2 = self.img_v2
        org_img = org_img[0]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        o_head_student1 = self.head(z1, z2)
        o_head_student2 = self.head(z2, z1)

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        o_head_teacher1 = self.teacher.head(zt1, zt2)
        o_head_teacher2 = self.teacher.head(zt2, zt1)

        teacher_loss1 = o_head_teacher1['cossim'].detach()
        teacher_loss2 = o_head_teacher2['cossim'].detach()

        student_loss1 = o_head_student1['cossim']
        student_loss2 = o_head_student2['cossim']

        # for plotting loss
        l_s1 = torch.mean(student_loss1)
        l_s2 = torch.mean(student_loss2)
        l_s = 1 / 2 * (l_s1 + l_s2).detach()
        l_t = 1 / 2 * (torch.mean(teacher_loss1) + torch.mean(teacher_loss2)).detach()

        index_lower1 = student_loss1 < teacher_loss1
        index_lower2 = student_loss2 < teacher_loss2

        index_higher1 = torch.bitwise_not(index_lower1)
        index_higher2 = torch.bitwise_not(index_lower2)

        # log frac
        num1 = float(torch.nonzero(index_lower1).size(0))
        num2 = float(torch.nonzero(index_lower2).size(0))
        total_number = float(img_v1.size(0))
        frac = (num1 + num2) / (total_number * 2)
        # put test here
        # log lpips
        d = self.fn_alex(img_v1, img_v2)

        # predictions
        p1 = o_head_student1['pred']
        p2 = o_head_student2['pred']

        pt1 = o_head_teacher1['pred']
        pt2 = o_head_teacher2['pred']


        # log image
        if self.save_images:
            images_lower1 = img_v1[index_lower1]
            images_lower1_proj = img_v2[index_lower1]
            images_higher1 = img_v1[index_higher1]
            images_higher2_proj = img_v2[index_higher1]
            ori_lower1 = org_img[index_lower1]
            ori_higher1 = org_img[index_higher1]

            set_1 = [images_lower1, images_lower1_proj, ori_lower1, student_loss1[index_lower1],
                     teacher_loss1[index_lower1], d[index_lower1],
                     images_higher1, images_higher2_proj, ori_higher1, student_loss1[index_higher1],
                     teacher_loss1[index_higher1], d[index_higher1],
                     p1[index_lower1], p2[index_lower1], pt1[index_lower1], pt2[index_lower1],
                     p1[index_higher1], p2[index_higher1], pt1[index_higher1], pt2[index_higher1]]

            # images_lower2 = img[0][index_lower2]
            # images_lower2_proj = img[1][index_lower2]
            # images_higher2 = img[0][index_higher2]
            # images_higher2_proj = img[1][index_higher2]
            # ori_lower2 = org_img[index_lower2]
            # ori_higher2 = org_img[index_higher2]
            #
            # set_2 = [images_lower2, images_lower2_proj, ori_lower2, student_loss2[index_lower2],
            #          teacher_loss2[index_lower2], d[index_lower2],
            #          images_higher2, images_higher2_proj, ori_higher2, student_loss2[index_higher2],
            #          teacher_loss2[index_higher2], d[index_higher2]]
            set_2 = None
        else:
            set_1, set_2 = None, None

        loss1 = torch.mean(student_loss1)
        loss2 = torch.mean(student_loss2)

        losses = 1 / 2 * (loss1 + loss2)
        return dict(loss=losses, l_student=l_s, l_teacher=l_t, frac=torch.tensor(frac).cuda()), set_1, set_2




@ALGORITHMS.register_module()
class SimDisKD_OLIH(SimSiamKD): # optimize lower, minimize Higher
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimDisKD_OLIH, self).__init__(backbone, neck, head, init_cfg, **kwargs)

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

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        p1 = self.head(z1, z2, loss_cal=False)
        p2 = self.head(z2, z1, loss_cal=False)

        pt1 = self.teacher.head(zt1, zt2, loss_cal=False).detach()
        pt2 = self.teacher.head(zt2, zt1, loss_cal=False).detach()

        teacher_loss1 = cosine_sim(pt1, zt2, mean=False).detach()
        teacher_loss2 = cosine_sim(pt2, zt1, mean=False).detach()

        student_loss1 = cosine_sim(p1, z2, mean=False)
        student_loss2 = cosine_sim(p2, z1, mean=False)
        # teacher_loss1 = self.teacher.head(zt1, zt2)['cossim'].detach()
        # teacher_loss2 = self.teacher.head(zt2, zt1)['cossim'].detach()
        #
        # student_loss1 = self.head(z1, z2)['cossim']
        # student_loss2 = self.head(z2, z1)['cossim']

        # for plotting loss
        l_s1 = torch.mean(student_loss1)
        l_s2 = torch.mean(student_loss2)
        l_s = 1/2 * (l_s1 + l_s2).detach()
        l_t = 1/2 * (torch.mean(teacher_loss1) + torch.mean(teacher_loss2)).detach()

        index_lower1 = student_loss1 < teacher_loss1
        index_lower2 = student_loss2 < teacher_loss2

        index_higher1 = torch.bitwise_not(index_lower1)
        index_higher2 = torch.bitwise_not(index_lower2)

        #loss 1
        #image1
        if student_loss1[index_higher1].shape[0] == 0:
            loss1_stl1 = 0.0
        else:
            loss1_stl1 = torch.mean(student_loss1[index_higher1]) * 0.0
        #image2
        if student_loss2[index_higher2].shape[0] == 0:
            loss1_stl2 = 0.0
        else:
            loss1_stl2 = torch.mean(student_loss2[index_higher2]) * 0.0
        loss1 = 0.5 * (loss1_stl1 + loss1_stl2) #loss1 here

        #loss 2
        #image1
        if student_loss1[index_lower1].shape[0] == 0:
            loss2_stl1 = 0.0
        else:
            loss2_stl1 = nn.functional.mse_loss(student_loss1[index_lower1], teacher_loss1[index_lower1])
        #image2
        if student_loss2[index_lower2].shape[0] == 0:
            loss2_stl2 = 0.0
        else:
            loss2_stl2 = nn.functional.mse_loss(student_loss2[index_lower2], teacher_loss2[index_lower2])
        loss2 = 0.5 * (loss2_stl1 + loss2_stl2) #loss2 here

        olih_losses = loss1 + loss2

        # SimDis loss
        distillation_loss1 = cosine_sim(p1, pt1)
        distillation_loss2 = cosine_sim(p2, pt2)
        distillation_losses = 0.5 * (distillation_loss1 + distillation_loss2)

        losses = olih_losses + distillation_losses

        return dict(loss=losses, l_student=l_s, l_teacher=l_t)


@ALGORITHMS.register_module()
class SimDisKD_OLMH(SimSiamKD): # optimize lower, minimize Higher
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimDisKD_OLMH, self).__init__(backbone, neck, head, init_cfg, **kwargs)

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

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        p1 = self.head(z1, z2, loss_cal=False)
        p2 = self.head(z2, z1, loss_cal=False)

        pt1 = self.teacher.head(zt1, zt2, loss_cal=False).detach()
        pt2 = self.teacher.head(zt2, zt1, loss_cal=False).detach()

        teacher_loss1 = cosine_sim(pt1, zt2, mean=False).detach()
        teacher_loss2 = cosine_sim(pt2, zt1, mean=False).detach()

        student_loss1 = cosine_sim(p1, z2, mean=False)
        student_loss2 = cosine_sim(p2, z1, mean=False)
        # teacher_loss1 = self.teacher.head(zt1, zt2)['cossim'].detach()
        # teacher_loss2 = self.teacher.head(zt2, zt1)['cossim'].detach()
        #
        # student_loss1 = self.head(z1, z2)['cossim']
        # student_loss2 = self.head(z2, z1)['cossim']

        # for plotting loss
        l_s1 = torch.mean(student_loss1)
        l_s2 = torch.mean(student_loss2)
        l_s = 1/2 * (l_s1 + l_s2).detach()
        l_t = 1/2 * (torch.mean(teacher_loss1) + torch.mean(teacher_loss2)).detach()

        index_lower1 = student_loss1 < teacher_loss1
        index_lower2 = student_loss2 < teacher_loss2

        index_higher1 = torch.bitwise_not(index_lower1)
        index_higher2 = torch.bitwise_not(index_lower2)

        #loss 1
        #image1
        if student_loss1[index_higher1].shape[0] == 0:
            loss1_stl1 = 0.0
        else:
            loss1_stl1 = torch.mean(student_loss1[index_higher1])
        #image2
        if student_loss2[index_higher2].shape[0] == 0:
            loss1_stl2 = 0.0
        else:
            loss1_stl2 = torch.mean(student_loss2[index_higher2])
        loss1 = 0.5 * (loss1_stl1 + loss1_stl2) #loss1 here

        #loss 2
        #image1
        if student_loss1[index_lower1].shape[0] == 0:
            loss2_stl1 = 0.0
        else:
            loss2_stl1 = nn.functional.mse_loss(student_loss1[index_lower1], teacher_loss1[index_lower1])
        #image2
        if student_loss2[index_lower2].shape[0] == 0:
            loss2_stl2 = 0.0
        else:
            loss2_stl2 = nn.functional.mse_loss(student_loss2[index_lower2], teacher_loss2[index_lower2])
        loss2 = 0.5 * (loss2_stl1 + loss2_stl2) #loss2 here

        olmh_losses = loss1 + loss2

        # SimDis loss
        distillation_loss1 = cosine_sim(p1, pt1)
        distillation_loss2 = cosine_sim(p2, pt2)
        distillation_losses = 0.5 * (distillation_loss1 + distillation_loss2)

        losses = olmh_losses + distillation_losses

        return dict(loss=losses, l_student=l_s, l_teacher=l_t)

@ALGORITHMS.register_module()
class SimDisKD_tracklowhigh(SimSiamKD): # optimize lower, minimize Higher
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimDisKD_tracklowhigh, self).__init__(backbone, neck, head, init_cfg, **kwargs)

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

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        p1 = self.head(z1, z2, loss_cal=False)
        p2 = self.head(z2, z1, loss_cal=False)

        pt1 = self.teacher.head(zt1, zt2, loss_cal=False).detach()
        pt2 = self.teacher.head(zt2, zt1, loss_cal=False).detach()

        teacher_loss1 = cosine_sim(pt1, zt2, mean=False).detach()
        teacher_loss2 = cosine_sim(pt2, zt1, mean=False).detach()

        student_loss1 = cosine_sim(p1, z2, mean=False)
        student_loss2 = cosine_sim(p2, z1, mean=False)
        # teacher_loss1 = self.teacher.head(zt1, zt2)['cossim'].detach()
        # teacher_loss2 = self.teacher.head(zt2, zt1)['cossim'].detach()
        #
        # student_loss1 = self.head(z1, z2)['cossim']
        # student_loss2 = self.head(z2, z1)['cossim']

        # for plotting loss
        l_s1 = torch.mean(student_loss1)
        l_s2 = torch.mean(student_loss2)
        l_s = 1/2 * (l_s1 + l_s2).detach()
        l_t = 1/2 * (torch.mean(teacher_loss1) + torch.mean(teacher_loss2)).detach()

        index_lower1 = student_loss1 < teacher_loss1
        index_lower2 = student_loss2 < teacher_loss2

        index_higher1 = torch.bitwise_not(index_lower1)
        index_higher2 = torch.bitwise_not(index_lower2)

        # log frac
        num1 = float(torch.nonzero(index_lower1).size(0))
        num2 = float(torch.nonzero(index_lower2).size(0))
        total_number = float(img_v1.size(0))
        frac = (num1 + num2) / (total_number * 2)

        # SimDis loss
        distillation_loss1 = cosine_sim(p1, pt1)
        distillation_loss2 = cosine_sim(p2, pt2)
        distillation_losses = 0.5 * (distillation_loss1 + distillation_loss2)

        losses = distillation_losses

        return dict(loss=losses, l_student=l_s, l_teacher=l_t, frac=torch.tensor(frac).cuda())