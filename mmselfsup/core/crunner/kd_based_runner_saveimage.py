# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import platform
import shutil
import time
import warnings
import torchvision.transforms as T

import torch

import mmcv
from mmcv.runner.base_runner import BaseRunner
from mmcv.runner.epoch_based_runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.utils import get_host_info
from PIL import Image, ImageFont, ImageDraw
import numpy as np

MEAN=np.asarray([0.485, 0.456, 0.406])
STD=np.asarray([0.229, 0.224, 0.225])


def save_log_images(set_n, folder, epoch):
    '''
    Args:
        set_n: set_1 or set_2. Including
        [images_lower1, images_lower1_proj, ori_lower1, student_loss1[index_lower1], teacher_loss1[index_lower1],
        images_higher1, images_higher2_proj, ori_higher1, student_loss1[index_higher1], teacher_loss1[index_higher1]]
        folder: folder used to save images
    Returns: 
    '''
    saved_folder = os.path.join(folder, "saved_images", "epoch_%d"%(epoch))
    os.makedirs(saved_folder, exist_ok=True)
    lower_info = set_n[0:5]
    higher_info = set_n[5:]


    num_low = lower_info[0].size(0)
    num_high = higher_info[0].size(0)
    saved_folder_low = os.path.join(saved_folder, "low_images")
    saved_folder_high = os.path.join(saved_folder, "high_images")
    os.makedirs(saved_folder_low, exist_ok=True)
    os.makedirs(saved_folder_high, exist_ok=True)
    font  = ImageFont.load_default()
    transform = T.Compose([T.Normalize((-1 * MEAN/STD), (1.0/STD)), T.ToPILImage()])
    
    for i in range(num_low):
        grid = lower_info[2].size(2)
        height = lower_info[1].size(2) # get the height of small image
        save_image = Image.new('RGB', (3 * lower_info[2].size(2), lower_info[2].size(3)), (250, 250, 250))
        image_lower = transform(lower_info[0][i])
        image_lower_proj = transform(lower_info[1][i])
        ori_lower = transform(lower_info[2][i])
        save_image.paste(ori_lower, (0,0))
        save_image.paste(image_lower, (grid + 1, 0))
        save_image.paste(image_lower_proj, (grid * 2, 0))
        caption = " Student: %f \n Teacher: %f\n"%(lower_info[3][i], lower_info[4][i])
        draw = ImageDraw.Draw(save_image)
        draw.text((grid + 3, height + 2), caption, font=font, fill="black")
        save_image.save(os.path.join(saved_folder_low, "image_set%d_%d.png"%(i, torch.cuda.current_device())))
        save_image.close()

    for i in range(num_high):
        grid = higher_info[2].size(2)
        height = higher_info[1].size(2) # get the height of the small images
        save_imageh = Image.new('RGB', (3 * higher_info[2].size(2), higher_info[2].size(3)), (250, 250, 250))
        image_higher = transform(higher_info[0][i])
        image_higher_proj = transform(higher_info[1][i])
        ori_higher = transform(higher_info[2][i])
        save_imageh.paste(ori_higher, (0, 0))
        save_imageh.paste(image_higher, (grid, 0))
        save_imageh.paste(image_higher_proj, (grid * 2, 0))
        caption = " Student: %f \n Teacher: %f\n" % (higher_info[3][i], higher_info[4][i])
        draw = ImageDraw.Draw(save_imageh)
        draw.text((grid + 3, height + 2), caption, font=font, fill="black")
        save_imageh.save(os.path.join(saved_folder_high, "image_set%d_%d.png"%(i, torch.cuda.current_device())))
        save_imageh.close()




@RUNNERS.register_module()
class KDBasedRunnerSaveImages(BaseRunner):
    """KD-based Runner.

    This runner train models epoch by epoch. For each epoch, the runner feed in the teacher model.
    """
    def __init__(self,
                 model,
                 t_model,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None
                 ):
        super(KDBasedRunnerSaveImages, self).__init__(
                                            model,
                                            batch_processor,
                                            optimizer,
                                            work_dir,
                                            logger,
                                            meta,
                                            max_iters,
                                            max_epochs)
        self.t_model = t_model

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs, set_1, set_2 = self.model.train_step(data_batch, self.optimizer, self.t_model, self.save_images,
                                            **kwargs)
            if self.save_images:
                save_log_images(set_1, self.work_dir, self.epoch)
                set_1, set_2 = None, None
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        self.save_images = torch.tensor([False]).cuda()
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            if self.epoch % 10 == 0:
                if i == 0:
                    self.save_images = torch.tensor([True]).cuda()
                else:
                    self.save_images = torch.tensor([False]).cuda()
            else:
                self.save_images = torch.tensor([False]).cuda()
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

