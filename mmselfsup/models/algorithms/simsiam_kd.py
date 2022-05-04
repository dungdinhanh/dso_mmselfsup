# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import time
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck, build_algorithm
from mmcv.runner.checkpoint import load_checkpoint
from .base import BaseModel
import torch
import torch.nn.functional as F

from collections import OrderedDict


@ALGORITHMS.register_module()
class SimSiamKD(BaseModel):
    """SimSiam.

    Implementation of `Exploring Simple Siamese Representation Learning
    <https://arxiv.org/abs/2011.10566>`_.
    The operation of fixing learning rate of predictor is in
    `core/hooks/simsiam_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiamKD, self).__init__(init_cfg)
        assert neck is not None
        self.encoder = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.backbone = self.encoder[0]
        self.neck = self.encoder[1]
        assert head is not None
        self.head = build_head(head)
        self.teacher = None
        # self.teacher = build_algorithm(teacher)
        # self.teacher.eval()
        # load_checkpoint(self.teacher, teacher_path, None, strict=False, revise_keys=[(r'^module.', '')])
        # print(self.teacher.state_dict())
        # exit(0)



    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

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

        teacher_loss1 = self.teacher.head(zt1, zt2)['cossim'].detach()
        teacher_loss2 = self.teacher.head(zt2, zt1)['cossim'].detach()


        losses = 0.5 * (nn.functional.mse_loss(self.head(z1, z2)['cossim'], teacher_loss1) +
                        nn.functional.mse_loss(self.head(z2, z1)['cossim'], teacher_loss2))
        return dict(loss=losses)

    def train_step(self, data, optimizer, teacher_model):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        if self.teacher is None:
            self.teacher = teacher_model
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        if isinstance(data['img'], list):
            num_samples = len(data['img'][0].data)
        else:
            num_samples = len(data['img'].data)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs

@ALGORITHMS.register_module()
class SimSiamKD_GT(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiamKD_GT, self).__init__(
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

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        teacher_loss1 = self.teacher.head(zt1, zt2)['cossim']
        teacher_loss2 = self.teacher.head(zt2, zt1)['cossim']

        student_output1 = self.head(z1, z2)
        student_output2 = self.head(z2, z1)

        loss_kd = 0.5 * (nn.functional.mse_loss(student_output1['cossim'], teacher_loss1) +
                        nn.functional.mse_loss(student_output2['cossim'], teacher_loss2))
        loss_student = 0.5 * (student_output1['loss'] + student_output2['loss'])
        losses = loss_kd + loss_student
        return dict(loss=losses)

@ALGORITHMS.register_module()
class SimSiamKD_wNeg(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiamKD_wNeg, self).__init__(
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
        neg_img = img[2]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC
        z3 = self.encoder(neg_img)[0]

        zt1 = self.teacher.encoder(img_v1)[0]
        # zt2 = self.teacher.encoder(img_v2)[0]
        zt3 = self.teacher.encoder(neg_img)[0]

        teacher_loss3 = self.teacher.head(zt1, zt3)['cossim'].detach()
        teacher_loss4 = self.teacher.head(zt3, zt1)['cossim'].detach()

        student_output1 = self.head(z1, z2)
        student_output2 = self.head(z2, z1)
        student_output3 = self.head(z1, z3)
        student_output4 = self.head(z3, z1)

        loss_kd = 0.5 * (nn.functional.mse_loss(student_output3['cossim'], teacher_loss3) +
                        nn.functional.mse_loss(student_output4['cossim'], teacher_loss4))
        loss_student = 0.5 * (student_output1['loss'] + student_output2['loss'])
        losses = loss_kd + loss_student
        return dict(loss=losses)


@ALGORITHMS.register_module()
class SimSiamKD_PoswNeg(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiamKD_PoswNeg, self).__init__(
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
        neg_img = img[2]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC
        z3 = self.encoder(neg_img)[0]

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]
        zt3 = self.teacher.encoder(neg_img)[0]

        teacher_loss1 = self.teacher.head(zt1, zt2)['cossim'].detach()
        teacher_loss2 = self.teacher.head(zt2, zt1)['cossim'].detach()
        teacher_loss3 = self.teacher.head(zt1, zt3)['cossim'].detach()
        teacher_loss4 = self.teacher.head(zt3, zt1)['cossim'].detach()

        student_output1 = self.head(z1, z2)
        student_output2 = self.head(z2, z1)
        student_output3 = self.head(z1, z3)
        student_output4 = self.head(z3, z1)

        loss_kd_neg = 0.5 * (nn.functional.mse_loss(student_output3['cossim'], teacher_loss3) +
                        nn.functional.mse_loss(student_output4['cossim'], teacher_loss4))

        loss_kd_pos = 0.5 * (nn.functional.mse_loss(student_output1['cossim'], teacher_loss1) +
                        nn.functional.mse_loss(student_output2['cossim'], teacher_loss2))
        # loss_student = 0.5 * (student_output1['loss'] + student_output2['loss'])
        losses = loss_kd_pos + loss_kd_neg
        return dict(loss=losses)

@ALGORITHMS.register_module()
class SimDis_PoswNeg(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimDis_PoswNeg, self).__init__(
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
        neg_img = img[2]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC
        z3 = self.encoder(neg_img)[0]

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]
        zt3 = self.teacher.encoder(neg_img)[0]

        p1 = self.head(z1, z2, loss_cal=False)
        p2 = self.head(z2, z1, loss_cal=False)
        # p3 = self.head(z1, z3, loss_cal=False)
        p3 = self.head(z3, z1, loss_cal=False)

        student_cs1 = cosine_sim(p1, z2, False)
        student_cs2 = cosine_sim(p2, z1, False)
        student_cs3 = cosine_sim(p1, z3, False)
        student_cs4 = cosine_sim(p3, z1, False)

        pt1 = self.teacher.head(zt1, zt2, loss_cal=False).detach()
        pt2 = self.teacher.head(zt2, zt1, loss_cal=False).detach()
        pt3 = self.teacher.head(zt3, zt1, loss_cal=False).detach()

        teacher_cs1 = cosine_sim(pt1, zt2, False).detach()
        teacher_cs2 = cosine_sim(pt2, zt1, False).detach()
        teacher_cs3 = cosine_sim(pt1, zt3, False).detach()
        teacher_cs4 = cosine_sim(pt3, zt1, False).detach()

        loss_kd_neg = 0.5 * (nn.functional.mse_loss(student_cs3, teacher_cs3) +
                        nn.functional.mse_loss(student_cs4, teacher_cs4))

        loss_kd_pos = 0.5 * (nn.functional.mse_loss(student_cs1, teacher_cs1) +
                        nn.functional.mse_loss(student_cs2, teacher_cs2))
        # loss_student = 0.5 * (student_output1['loss'] + student_output2['loss'])

        distillation_loss1 = cosine_sim(p1, pt2) + cosine_sim(p1, zt2)
        distillation_loss2 = cosine_sim(p2, pt1) + cosine_sim(p2, zt1)

        distillation_loss = 0.5 * (distillation_loss1 + distillation_loss2)

        simsiam_losses = loss_kd_pos + loss_kd_neg
        losses = simsiam_losses + distillation_loss
        return dict(loss=losses)

@ALGORITHMS.register_module()
class SimDis_Pos(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimDis_Pos, self).__init__(
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
        neg_img = img[2]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC
        z3 = self.encoder(neg_img)[0]

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]
        zt3 = self.teacher.encoder(neg_img)[0]

        p1 = self.head(z1, z2, loss_cal=False)
        p2 = self.head(z2, z1, loss_cal=False)
        # p3 = self.head(z1, z3, loss_cal=False)
        p3 = self.head(z3, z1, loss_cal=False)

        student_cs1 = cosine_sim(p1, z2, False)
        student_cs2 = cosine_sim(p2, z1, False)
        student_cs3 = cosine_sim(p1, z3, False)
        student_cs4 = cosine_sim(p3, z1, False)

        pt1 = self.teacher.head(zt1, zt2, loss_cal=False).detach()
        pt2 = self.teacher.head(zt2, zt1, loss_cal=False).detach()
        pt3 = self.teacher.head(zt3, zt1, loss_cal=False).detach()

        teacher_cs1 = cosine_sim(pt1, zt2, False).detach()
        teacher_cs2 = cosine_sim(pt2, zt1, False).detach()
        teacher_cs3 = cosine_sim(pt1, zt3, False).detach()
        teacher_cs4 = cosine_sim(pt3, zt1, False).detach()

        # loss_kd_neg = 0.5 * (nn.functional.mse_loss(student_cs3, teacher_cs3) +
        #                 nn.functional.mse_loss(student_cs4, teacher_cs4))

        loss_kd_pos = 0.5 * (nn.functional.mse_loss(student_cs1, teacher_cs1) +
                        nn.functional.mse_loss(student_cs2, teacher_cs2))
        # loss_student = 0.5 * (student_output1['loss'] + student_output2['loss'])

        distillation_loss1 = cosine_sim(p1, pt2) + cosine_sim(p1, zt2)
        distillation_loss2 = cosine_sim(p2, pt1) + cosine_sim(p2, zt1)

        distillation_loss = 0.5 * (distillation_loss1 + distillation_loss2)

        simsiam_losses = loss_kd_pos
        losses = simsiam_losses + distillation_loss
        return dict(loss=losses)

@ALGORITHMS.register_module()
class SimDis_wNeg(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimDis_wNeg, self).__init__(
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
        neg_img = img[2]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC
        z3 = self.encoder(neg_img)[0]

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]
        zt3 = self.teacher.encoder(neg_img)[0]

        p1 = self.head(z1, z2, loss_cal=False)
        p2 = self.head(z2, z1, loss_cal=False)
        # p3 = self.head(z1, z3, loss_cal=False)
        p3 = self.head(z3, z1, loss_cal=False)

        student_cs1 = cosine_sim(p1, z2, False)
        student_cs2 = cosine_sim(p2, z1, False)
        student_cs3 = cosine_sim(p1, z3, False)
        student_cs4 = cosine_sim(p3, z1, False)

        pt1 = self.teacher.head(zt1, zt2, loss_cal=False).detach()
        pt2 = self.teacher.head(zt2, zt1, loss_cal=False).detach()
        pt3 = self.teacher.head(zt3, zt1, loss_cal=False).detach()

        teacher_cs1 = cosine_sim(pt1, zt2, False).detach()
        teacher_cs2 = cosine_sim(pt2, zt1, False).detach()
        teacher_cs3 = cosine_sim(pt1, zt3, False).detach()
        teacher_cs4 = cosine_sim(pt3, zt1, False).detach()

        loss_kd_neg = 0.5 * (nn.functional.mse_loss(student_cs3, teacher_cs3) +
                        nn.functional.mse_loss(student_cs4, teacher_cs4))

        # loss_kd_pos = 0.5 * (nn.functional.mse_loss(student_cs1, teacher_cs1) +
        #                 nn.functional.mse_loss(student_cs2, teacher_cs2))
        loss_siam = 0.5 * (student_cs1.mean() + student_cs2.mean())

        distillation_loss1 = cosine_sim(p1, pt2) + cosine_sim(p1, zt2)
        distillation_loss2 = cosine_sim(p2, pt1) + cosine_sim(p2, zt1)

        distillation_loss = 0.5 * (distillation_loss1 + distillation_loss2)

        imp_losses = loss_siam + loss_kd_neg
        losses = imp_losses + distillation_loss
        return dict(loss=losses)

@ALGORITHMS.register_module()
class SimSiamKDZT(SimSiamKD):
    """
    Simsiam KD with Z from teacher
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiamKDZT, self).__init__(
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

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        # teacher_loss1 = self.teacher.head(zt1, zt2)['loss']
        # teacher_loss2 = self.teacher.head(zt2, zt1)['loss']

        losses = 0.5 * (self.head(z1, zt2)['loss'] + self.head(z2, zt1)['loss'])
        return dict(loss=losses)


def kdloss(y, teacher_scores, temperature=3):
    """
    Loss used for previous KD experiments
    """
    p = F.log_softmax(y / temperature, dim=1)
    q = F.softmax(teacher_scores / temperature, dim=1)
    l_kl = F.kl_div(p, q, reduction='batchmean')
    return l_kl

@ALGORITHMS.register_module()
class SimSiamKD_PredMatching(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 teacher_project=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiamKD_PredMatching, self).__init__(
                                                     backbone,
                                                     neck=neck,
                                                     head=head,
                                                     init_cfg=init_cfg,
                                                        **kwargs)
        self.projector = build_neck(teacher_project)

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

        # teacher_loss1 = self.teacher.head(zt1, zt2)['loss']
        # teacher_loss2 = self.teacher.head(zt2, zt1)['loss']
        output_head1 = self.head(z1, z2)
        output_head2 = self.head(z2, z1)

        output_thead1 = self.teacher.head(zt1, zt2)['pred']
        output_thead2 = self.teacher.head(zt2, zt1)['pred']

        output_project1 = self.projector((output_head1['pred'],))
        output_project2 = self.projector((output_head2['pred'],))

        loss_simsiam = 0.5 * (output_head1['loss'] + output_head2['loss'])
        loss_kd = kdloss(output_project1[0], output_thead1) + kdloss(output_project2[0], output_thead2)
        losses = 0.5 * loss_kd + 0.5 * loss_simsiam
        return dict(loss=losses)


@ALGORITHMS.register_module()
class SimDis_Siam_simplified(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimDis_Siam_simplified, self).__init__(
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

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        p1 = self.head(z1, z2, loss_cal=False)
        p2 = self.head(z2, z1, loss_cal=False)

        pt1 = self.teacher.head(zt1, zt2, loss_cal=False)
        pt2 = self.teacher.head(zt2, zt1, loss_cal=False)

        simsiam_loss = 0.5 * (cosine_sim(p1, z2) + cosine_sim(p2, z1))
        distillation_loss1 = cosine_sim(p1, pt2) + cosine_sim(p1, zt2)
        distillation_loss2 = cosine_sim(p2, pt1) + cosine_sim(p2, zt1)

        distillation_loss = 0.5 * (distillation_loss1 + distillation_loss2)

        losses = simsiam_loss + distillation_loss

        return dict(loss=losses)


def cosine_sim(input, target, mean=True):
    target = target.detach()
    pred_norm = nn.functional.normalize(input, dim=1)
    target_norm = nn.functional.normalize(target, dim=1)
    cs_sim = -(pred_norm * target_norm).sum(dim=1)
    if mean:
        loss = cs_sim.mean()
        return loss
    else:
        return cs_sim