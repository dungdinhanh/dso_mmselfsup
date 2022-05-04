# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.utils import build_from_cfg
from torchvision.transforms import Compose
from torchvision import transforms as T

from .base import BaseDataset
from .builder import DATASETS, PIPELINES, build_datasource
from .utils import to_numpy


@DATASETS.register_module()
class MultiViewDataset(BaseDataset):
    """The dataset outputs multiple views of an image.

    The number of views in the output dict depends on `num_views`. The
    image can be processed by one pipeline or multiple piepelines.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        num_views (list): The number of different views.
        pipelines (list[list[dict]]): A list of pipelines, where each pipeline
            contains elements that represents an operation defined in
            `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.

    Examples:
        >>> dataset = MultiViewDataset(data_source, [2], [pipeline])
        >>> output = dataset[idx]
        The output got 2 views processed by one pipeline.

        >>> dataset = MultiViewDataset(
        >>>     data_source, [2, 6], [pipeline1, pipeline2])
        >>> output = dataset[idx]
        The output got 8 views processed by two pipelines, the first two views
        were processed by pipeline1 and the remaining views by pipeline2.
    """

    def __init__(self, data_source, num_views, pipelines, prefetch=False):
        assert len(num_views) == len(pipelines)
        self.data_source = build_datasource(data_source)
        self.pipelines = []
        for pipe in pipelines:
            pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            self.pipelines.append(pipeline)
        self.prefetch = prefetch

        trans = []
        assert isinstance(num_views, list)
        for i in range(len(num_views)):
            trans.extend([self.pipelines[i]] * num_views[i])
        self.trans = trans

    def __getitem__(self, idx):
        img = self.data_source.get_img(idx)
        multi_views = list(map(lambda trans: trans(img), self.trans))
        if self.prefetch:
            multi_views = [
                torch.from_numpy(to_numpy(img)) for img in multi_views
            ]
        return dict(img=multi_views)

    def evaluate(self, results, logger=None):
        return NotImplemented


import random


@DATASETS.register_module()
class MultiViewDatasetwNegative(MultiViewDataset):
    """The dataset outputs multiple views of an image, and a view of a different batch.

    The number of views in the output dict depends on `num_views`. The
    image can be processed by one pipeline or multiple piepelines.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        num_views (list): The number of different views.
        pipelines (list[list[dict]]): A list of pipelines, where each pipeline
            contains elements that represents an operation defined in
            `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.

    Examples:
        >>> dataset = MultiViewDataset(data_source, [2], [pipeline])
        >>> output = dataset[idx]
        The output got 2 views processed by one pipeline.

        >>> dataset = MultiViewDataset(
        >>>     data_source, [2, 6], [pipeline1, pipeline2])
        >>> output = dataset[idx]
        The output got 8 views processed by two pipelines, the first two views
        were processed by pipeline1 and the remaining views by pipeline2.
    """

    def __init__(self, data_source, num_views, pipelines, prefetch=False):
        super(MultiViewDatasetwNegative, self).__init__(data_source, num_views, pipelines, prefetch)
        self.num_images = len(self.data_source)
        print("Total number of images: %d"%self.num_images)

    def __getitem__(self, idx):
        img = self.data_source.get_img(idx)
        diff_idx = random.randint(0, self.num_images-1)
        if diff_idx == idx:
            if diff_idx == 0:
                diff_idx += 1
            else:
                diff_idx -= 1
        neg_img = self.data_source.get_img(diff_idx)
        multi_views = list(map(lambda trans: trans(img), self.trans))
        neg_views = list(map(lambda pipelines: pipelines(neg_img), self.pipelines))
        all_views = multi_views + neg_views
        if self.prefetch:
            all_views = [
                torch.from_numpy(to_numpy(img)) for img in all_views
            ]

        return dict(img=all_views)

    def evaluate(self, results, logger=None):
        return NotImplemented

@DATASETS.register_module()
class MultiViewDatasetwOriginal(MultiViewDataset):
    """The dataset outputs multiple views of an image, and a view of a different batch.

    The number of views in the output dict depends on `num_views`. The
    image can be processed by one pipeline or multiple piepelines.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        num_views (list): The number of different views.
        pipelines (list[list[dict]]): A list of pipelines, where each pipeline
            contains elements that represents an operation defined in
            `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.

    Examples:
        >>> dataset = MultiViewDataset(data_source, [2], [pipeline])
        >>> output = dataset[idx]
        The output got 2 views processed by one pipeline.

        >>> dataset = MultiViewDataset(
        >>>     data_source, [2, 6], [pipeline1, pipeline2])
        >>> output = dataset[idx]
        The output got 8 views processed by two pipelines, the first two views
        were processed by pipeline1 and the remaining views by pipeline2.
    """

    def __init__(self, data_source, num_views, pipelines, prefetch=False):
        super(MultiViewDatasetwOriginal, self).__init__(data_source, num_views, pipelines, prefetch)
        self.num_images = len(self.data_source)
        print("Total number of images: %d"%self.num_images)
        self.ori_transform = Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]),
                                      T.CenterCrop(size=256)]
                                     )
    def __getitem__(self, idx):
        img = self.data_source.get_img(idx)
        multi_views = list(map(lambda trans: trans(img), self.trans))
        img_org = self.ori_transform(self.data_source.get_img(idx))
        all_views = multi_views
        if self.prefetch:
            all_views = [
                torch.from_numpy(to_numpy(img)) for img in multi_views
            ] + [torch.from_numpy(to_numpy(img_org))]

        return dict(img=all_views, org_img=[img_org])

    def evaluate(self, results, logger=None):
        return NotImplemented

@DATASETS.register_module()
class MultiViewDatasetwOriginalLog(MultiViewDataset):
    """The dataset outputs multiple views of an image, and a view of a different batch.

    The number of views in the output dict depends on `num_views`. The
    image can be processed by one pipeline or multiple piepelines.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        num_views (list): The number of different views.
        pipelines (list[list[dict]]): A list of pipelines, where each pipeline
            contains elements that represents an operation defined in
            `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.

    Examples:
        >>> dataset = MultiViewDataset(data_source, [2], [pipeline])
        >>> output = dataset[idx]
        The output got 2 views processed by one pipeline.

        >>> dataset = MultiViewDataset(
        >>>     data_source, [2, 6], [pipeline1, pipeline2])
        >>> output = dataset[idx]
        The output got 8 views processed by two pipelines, the first two views
        were processed by pipeline1 and the remaining views by pipeline2.
    """

    def __init__(self, data_source, num_views, pipelines, prefetch=False):
        super(MultiViewDatasetwOriginalLog, self).__init__(data_source, num_views, pipelines, prefetch)
        self.num_images = len(self.data_source)
        print("Total number of images: %d"%self.num_images)
        self.ori_transform = Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]),
                                      T.CenterCrop(size=256)]
                                     )
    def __getitem__(self, idx):
        img = self.data_source.get_img(idx)
        multi_views = list(map(lambda trans: trans(img), self.trans))
        img_org = self.ori_transform(self.data_source.get_img(idx))
        all_views = multi_views
        if self.prefetch:
            all_views = [
                torch.from_numpy(to_numpy(img)) for img in multi_views
            ] + [torch.from_numpy(to_numpy(img_org))]

        return dict(img=all_views, org_img=[img_org])

    def evaluate(self, results, logger=None):
        return NotImplemented