# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck, build_algorithm
from mmcv.runner.checkpoint import load_checkpoint
from .base import BaseModel
import torch


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

        # teacher_loss1 = self.teacher.head(zt1, zt2)['loss']
        # teacher_loss2 = self.teacher.head(zt2, zt1)['loss']

        losses = 0.5 * (self.head(z1, zt2)['loss'] + self.head(z2, zt1)['loss'])
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
