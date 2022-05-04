# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel
from mmcv.runner import BaseModule, auto_fp16
from .utils import *


@ALGORITHMS.register_module()
class SimSiam(BaseModel):
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
        super(SimSiam, self).__init__(init_cfg)
        assert neck is not None
        self.encoder = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.backbone = self.encoder[0]
        self.neck = self.encoder[1]
        assert head is not None
        self.head = build_head(head)

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
        img_v1 = img[0]
        img_v2 = img[1]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        losses = 0.5 * (self.head(z1, z2)['loss'] + self.head(z2, z1)['loss'])
        return dict(loss=losses)


@ALGORITHMS.register_module()
class SimSiamL(BaseModel):
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
        super(SimSiamL, self).__init__(init_cfg)
        assert neck is not None
        self.encoder = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.backbone = self.encoder[0]
        self.neck = self.encoder[1]
        assert head is not None
        self.head = build_head(head)

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
        img_v1 = img[0]
        img_v2 = img[1]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        losses = 0.5 * (self.head(z1, z2)['loss'] + self.head(z2, z1)['loss'])
        return dict(loss=losses, l_student=losses)


@ALGORITHMS.register_module()
class SimSiamVis(BaseModel):
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
        super(SimSiamVis, self).__init__(init_cfg)
        assert neck is not None
        self.encoder = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.backbone = self.encoder[0]
        self.neck = self.encoder[1]
        assert head is not None
        self.head = build_head(head)
        self.teacher = None

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

    def extract_encoder(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.encoder(img)
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
        img_v1 = img[0]
        img_v2 = img[1]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC
        losses = 0.5 * (self.head(z1, z2)['loss'] + self.head(z2, z1)['loss'])
        return dict(loss=losses)

    def extract_diff(self, img):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.
        Returns:
            loss[str, Tensor]: A dictionary of loss components
        """

        z = self.encoder(img)[0]
        p = self.head(z, z, loss_cal=False)
        x = p - z
        return x

    def extract_diff_multiview(self, img):
        assert isinstance(img, list)
        img_v1 = img[0]
        img_v2 = img[1]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        p1 = self.head(z1, z2, loss_cal=False)
        p1 = nn.functional.normalize(p1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        return p1 - z2

    def lower_teacher(self, img):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.
        Returns:
            loss[str, Tensor]: A dictionary of loss components
        """
        assert isinstance(img, list)
        img_v1 = img[0]
        img_v2 = img[1]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        student_loss1 = self.head(z1, z2)['loss']

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        teacher_loss1 = self.teacher.head(zt1, zt2)['loss']

        lower_indices = student_loss1 < teacher_loss1
        print(lower_indices)
        exit(0)

        return None


    @auto_fp16(apply_to=('img',))
    def forward(self, img, mode='train', teacher=None, **kwargs):
        """Forward function of model.

        Calls either forward_train, forward_test or extract_feat function
        according to the mode.
        """
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.extract_feat(img)
        elif mode == 'extract_encoder':
            return self.extract_encoder(img)
        elif mode == 'extract_diff':
            return self.extract_diff(img)
        elif mode == 'extract_diff_multiview':
            return self.extract_diff_multiview(img)
        elif mode == 'lower_teacher':
            if teacher is None:
                print("Must include teacher weights and configs")
                exit(0)
            if self.teacher is None:
                self.teacher = teacher
            return self.lower_teacher(img, **kwargs)
        else:
            raise Exception(f'No such mode: {mode}')


@ALGORITHMS.register_module()
class SimSiamLogMin(SimSiam):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiamLogMin, self).__init__(backbone=backbone,
                                            neck=neck,
                                            head=head,
                                            init_cfg=init_cfg)

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
        img_v1 = img[0]
        img_v2 = img[1]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC
        output1 = self.head(z1, z2, loss_cal=False)
        output2 = self.head(z2, z1, loss_cal=False)

        loss1 = cosine_sim(output1, z2, False)
        loss2 = cosine_sim(output2, z1, False)

        mean_loss1 = loss1.mean()
        mean_loss2 = loss2.mean()

        min_loss1 = loss1.min()
        min_loss2 = loss2.min()

        losses = 0.5 * (mean_loss1 + mean_loss2)

        return dict(loss=losses), [min_loss1, min_loss2]

    def train_step(self, data, optimizer):
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
        losses, log_min = self(**data)
        loss, log_vars = self._parse_losses(losses)

        if isinstance(data['img'], list):
            num_samples = len(data['img'][0].data)
        else:
            num_samples = len(data['img'].data)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs, log_min