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
from mmcv.runner.hooks import HOOKS, Hook
import numpy as np
import matplotlib.pyplot as plt

MEAN=np.asarray([0.485, 0.456, 0.406])
STD=np.asarray([0.229, 0.224, 0.225])


def save_log_images(set_n, folder, epoch, iteration):
    '''
    Args:
        set_n: set_1 or set_2. Including
        [images_lower1, images_lower1_proj, ori_lower1, student_loss1[index_lower1], teacher_loss1[index_lower1],
        images_higher1, images_higher2_proj, ori_higher1, student_loss1[index_higher1], teacher_loss1[index_higher1]]
        folder: folder used to save images
    Returns: 
    '''
    saved_folder = os.path.join(folder, "saved_images", "epoch_%d_iter_%d"%(epoch, iteration))
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


def save_log_images_lpips(set_n, folder, epoch, iteration):
    '''
    Args:
        set_n: set_1 or set_2. Including
        [images_lower1, images_lower1_proj, ori_lower1, student_loss1[index_lower1], teacher_loss1[index_lower1],
        images_higher1, images_higher2_proj, ori_higher1, student_loss1[index_higher1], teacher_loss1[index_higher1]]
        folder: folder used to save images
    Returns:
    '''
    saved_folder = os.path.join(folder, "saved_images", "epoch_%d_iter_%d" % (epoch, iteration))
    os.makedirs(saved_folder, exist_ok=True)
    lower_info = set_n[0:6]
    higher_info = set_n[6:12]

    num_low = lower_info[0].size(0)
    num_high = higher_info[0].size(0)
    saved_folder_low = os.path.join(saved_folder, "low_images")
    saved_folder_high = os.path.join(saved_folder, "high_images")
    os.makedirs(saved_folder_low, exist_ok=True)
    os.makedirs(saved_folder_high, exist_ok=True)
    font = ImageFont.load_default()
    transform = T.Compose([T.Normalize((-1 * MEAN / STD), (1.0 / STD)), T.ToPILImage()])

    for i in range(num_low):
        grid = lower_info[2].size(2)
        height = lower_info[1].size(2)  # get the height of small image
        save_image = Image.new('RGB', (3 * lower_info[2].size(2), lower_info[2].size(3)), (250, 250, 250))
        image_lower = transform(lower_info[0][i])
        image_lower_proj = transform(lower_info[1][i])
        ori_lower = transform(lower_info[2][i])
        save_image.paste(ori_lower, (0, 0))
        save_image.paste(image_lower, (grid + 1, 0))
        save_image.paste(image_lower_proj, (grid * 2, 0))
        caption = " Student: %f \t lpips distance: %f \n Teacher: %f\n" % (lower_info[3][i], lower_info[5][i], lower_info[4][i])
        draw = ImageDraw.Draw(save_image)
        draw.text((grid + 3, height + 2), caption, font=font, fill="black")
        save_image.save(os.path.join(saved_folder_low, "image_set%d_%d.png" % (i, torch.cuda.current_device())))

    for i in range(num_high):
        grid = higher_info[2].size(2)
        height = higher_info[1].size(2)  # get the height of the small images
        save_imageh = Image.new('RGB', (3 * higher_info[2].size(2), higher_info[2].size(3)), (250, 250, 250))
        image_higher = transform(higher_info[0][i])
        image_higher_proj = transform(higher_info[1][i])
        ori_higher = transform(higher_info[2][i])
        save_imageh.paste(ori_higher, (0, 0))
        save_imageh.paste(image_higher, (grid, 0))
        save_imageh.paste(image_higher_proj, (grid * 2, 0))
        caption = " Student: %f \t lpips distance: %f \n Teacher: %f\n" % (higher_info[3][i], higher_info[5][i] ,higher_info[4][i])
        draw = ImageDraw.Draw(save_imageh)
        draw.text((grid + 3, height + 2), caption, font=font, fill="black")
        save_imageh.save(os.path.join(saved_folder_high, "image_set%d_%d.png" % (i, torch.cuda.current_device())))

def save_statistics(set_n, folder):
    #temp_set = [student_loss_lower, # 0
                  # teacher_loss_lower, # 1
                  # lpips_lower, # 2
                  # student_loss_higher, # 3
                  # teacher_loss_higher, # 4
                  # lpips_higher, # 5
                  # p1_lower.cpu().detach().numpy(), # 6
                  # p2_lower.cpu().detach().numpy(), # 7
                  # pt1_lower.cpu().detach().numpy(), # 8
                  # pt2_lower.cpu().detach().numpy(), # 9
                  # p1_higher.cpu().detach().numpy(), # 10
                  # p2_higher.cpu().detach().numpy(), # 11
                  # pt1_higher.cpu().detach().numpy(), # 12
                  # pt2_higher.cpu().detach().numpy()] #13
    print("Start saving")
    saved_folder = os.path.join(folder, "statistics")
    os.makedirs(saved_folder, exist_ok=True)
    # save average, std lpips lower
    lpips_lower = set_n[2]
    avg_lpips_lower = np.mean(lpips_lower)
    std_lpips_lower = np.std(lpips_lower)

    print("Complete lpips lower")
    # save average, std lpips higher
    lpips_higher = set_n[5]
    avg_lpips_higher = np.mean(lpips_higher)
    std_lpips_higher = np.std(lpips_higher)

    print("Complete lpips higher")
    # save average, std student cs lower
    cs_student_lower = set_n[0]
    avg_cs_student_lower = np.mean(cs_student_lower)
    std_cs_student_lower = np.std(cs_student_lower)

    print("Complete cs student lower")

    # save average, std student cs higher
    cs_student_higher = set_n[3]
    avg_cs_student_higher = np.mean(cs_student_higher)
    std_cs_student_higher = np.std(cs_student_higher)

    print("Complete cs student higher")

    # save average, std teacher cs lower
    cs_teacher_lower = set_n[1]
    avg_cs_teacher_lower = np.mean(cs_teacher_lower)
    std_cs_teacher_lower = np.std(cs_teacher_lower)

    print("Complete cs teacher lower")
    # save average, std teacher cs higher
    cs_teacher_higher = set_n[4]
    avg_cs_teacher_higher = np.mean(cs_teacher_higher)
    std_cs_teacher_higher = np.std(cs_teacher_higher)

    print("Complete cs teacher higher")
    # save all statistics to file here
    file_name = os.path.join(saved_folder, "statistic.csv")
    f = open(file_name, "w")
    f.write("Model, Lpips lower, , Lpips higher ,CS lower, , CS  higher, ,\n")
    f.write(" , AVG, STD, AVG, STD, AVG, STD, AVG, STD, \n")
    f.write("SimSiam PosKD, %f, %f, %f, %f, %f, %f, %f, %f, \n"%(avg_lpips_lower, std_lpips_lower,
                                                             avg_lpips_higher, std_lpips_higher,
                                                             avg_cs_student_lower, std_cs_student_lower,
                                                             avg_cs_student_higher, std_cs_student_higher))

    f.write("SimSiam R50, %f, %f, %f, %f, %f, %f, %f, %f,\n"%(avg_lpips_lower, std_lpips_lower,
                                                             avg_lpips_higher, std_lpips_higher,
                                                              avg_cs_teacher_lower, std_cs_teacher_lower,
                                                              avg_cs_teacher_higher, std_cs_teacher_higher))

    f.close()
    print("Complete saving statistics")
    # save svd cov student lower
    cov_features_student_lower = np.cov(np.concatenate((set_n[6], set_n[7]), axis=0))
    _, s_student_lower, _ = np.linalg.svd(cov_features_student_lower)
    log_s_student_lower = np.log(s_student_lower)
    file_student_lower_svdcov = os.path.join(saved_folder, "svdcov_student_lower.png")
    plt.plot(log_s_student_lower)
    plt.savefig(file_student_lower_svdcov)
    plt.close()

    print("Complete saving svd cv student lower")
    # save svd cov student higher
    cov_features_student_higher = np.cov(np.concatenate((set_n[10], set_n[11]), axis=0))
    _, s_student_higher, _ = np.linalg.svd(cov_features_student_higher)
    log_s_student_higher = np.log(s_student_higher)
    file_student_higher_svdcov = os.path.join(saved_folder, "svdcov_student_higher.png")
    plt.plot(log_s_student_higher)
    plt.savefig(file_student_higher_svdcov)
    plt.close()

    print("Compelte saving svd cv student higher")

    # save svd cov teacher lower
    cov_features_teacher_lower = np.cov(np.concatenate((set_n[8], set_n[9]), axis=0))
    _, s_teacher_lower, _ = np.linalg.svd(cov_features_teacher_lower)
    log_s_teacher_lower = np.log(s_teacher_lower)
    file_teacher_lower_svdcov = os.path.join(saved_folder, 'svdcov_teacher_lower.png')
    plt.plot(log_s_teacher_lower)
    plt.savefig(file_teacher_lower_svdcov)
    plt.close()

    print("Complete saving svd cov teacher lower")

    # save svd cov teacher higher
    cov_features_teacher_higher = np.cov(np.concatenate((set_n[12], set_n[13]), axis=0))
    _, s_teacher_higher, _ = np.linalg.svd(cov_features_teacher_higher)
    log_s_teacher_higher = np.log(s_teacher_higher)
    file_teacher_higher_svdcov = os.path.join(saved_folder, 'svdcov_teacher_higher.png')
    plt.plot(log_s_teacher_higher)
    plt.savefig(file_teacher_higher_svdcov)
    plt.close()

    print("complete saving svd cov teacher higher")





@RUNNERS.register_module()
class KDBasedRunnerSaveImagesAll(BaseRunner):
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
        super(KDBasedRunnerSaveImagesAll, self).__init__(
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
                save_log_images(set_1, self.work_dir, self.epoch, self._inner_iter)
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
        self.save_images = torch.tensor([True]).cuda()
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
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
        self._epoch += 1

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

        while self.epoch < 1:
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

    def register_optimizer_hook(self, optimizer_config):
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'NoneOptimizerHook')
            hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook, priority='ABOVE_NORMAL')


@RUNNERS.register_module()
class KDBasedRunnerSaveImagesAllLPIPS(BaseRunner):
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
        super(KDBasedRunnerSaveImagesAllLPIPS, self).__init__(
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
                save_log_images_lpips(set_1, self.work_dir, self.epoch, self._inner_iter)
                set_1, set_2 = None, None
        else:
            outputs, set_1, set_2 = self.model.val_step(data_batch, self.optimizer, self.t_model, self.save_images,
                                            **kwargs)
            if self.save_images:
                save_log_images_lpips(set_1, self.work_dir, self.epoch, self._inner_iter)
                set_1, set_2 = None, None
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
        self.save_images = torch.tensor([True]).cuda()
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
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
        self.save_images = torch.tensor([True]).cuda()
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')
        self._epoch += 1

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

        while self.epoch < 1:
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
                    if mode == 'val' and self.epoch >= self._max_epochs:
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

    # def register_optimizer_hook(self, optimizer_config):
    #     if optimizer_config is None:
    #         return
    #     if isinstance(optimizer_config, dict):
    #         optimizer_config.setdefault('type', 'NoneOptimizerHook')
    #         hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
    #         print("GO INTO IF")
    #     else:
    #         optimizer_config = dict(lr=0.0, paramwise_options={'predictor': dict(fix_lr=True)})
    #         optimizer_config.setdefault('type', 'NoneOptimizerHook')
    #         hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
    #         print("GO INTO ELSE")
    #     self.register_hook(hook, priority='ABOVE_NORMAL')



@RUNNERS.register_module()
class KDBasedRunnerSaveImagesAllLPIPS2(BaseRunner):
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
        super(KDBasedRunnerSaveImagesAllLPIPS2, self).__init__(
                                                        model,
                                                        batch_processor,
                                                        optimizer,
                                                        work_dir,
                                                        logger,
                                                        meta,
                                                        max_iters,
                                                        max_epochs)
        self.t_model = t_model
        self.set_logs = None

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs, set_1, set_2 = self.model.train_step(data_batch, self.optimizer, self.t_model, self.save_images,
                                            **kwargs)
            if self.save_images:
                save_log_images_lpips(set_1, self.work_dir, self.epoch, self._inner_iter)
                set_1, set_2 = None, None
        else:
            outputs, set_1, set_2 = self.model.val_step(data_batch, self.optimizer, self.t_model, self.save_images,
                                            **kwargs)
            if self.save_images:
                save_log_images_lpips(set_1, self.work_dir, self.epoch, self._inner_iter)
                self.long_term_save(set_1)
                set_1, set_2 = None, None
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def long_term_save(self, set_logs):
        student_loss_lower = set_logs[3].cpu().detach().numpy()
        teacher_loss_lower = set_logs[4].cpu().detach().numpy()
        lpips_lower = set_logs[5].cpu().detach().numpy()

        student_loss_higher = set_logs[9].cpu().detach().numpy()
        teacher_loss_higher = set_logs[10].cpu().detach().numpy()
        lpips_higher = set_logs[11].cpu().detach().numpy()

        p1_lower, p2_lower, pt1_lower, pt2_lower = set_logs[12:16]
        p1_higher, p2_higher, pt1_higher, pt2_higher = set_logs[16:]
        temp_set = [student_loss_lower, # 0
                  teacher_loss_lower, # 1
                  lpips_lower, # 2
                  student_loss_higher, # 3
                  teacher_loss_higher, # 4
                  lpips_higher, # 5
                  p1_lower.cpu().detach().numpy(), # 6
                  p2_lower.cpu().detach().numpy(), # 7
                  pt1_lower.cpu().detach().numpy(), # 8
                  pt2_lower.cpu().detach().numpy(), # 9
                  p1_higher.cpu().detach().numpy(), # 10
                  p2_higher.cpu().detach().numpy(), # 11
                  pt1_higher.cpu().detach().numpy(), # 12
                  pt2_higher.cpu().detach().numpy()] #13

        if self.set_logs is None:
            self.set_logs = temp_set
        else:
            for i in range(len(self.set_logs)):
                self.set_logs[i] = np.concatenate((self.set_logs[i], temp_set[i]), axis=0)

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        self.save_images = torch.tensor([True]).cuda()
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
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
        self.save_images = torch.tensor([True]).cuda()
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')
        self._epoch += 1

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

        while self.epoch < 1:
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
                    if mode == 'val' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)
        save_statistics(self.set_logs, self.work_dir)
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

    # def register_optimizer_hook(self, optimizer_config):
    #     if optimizer_config is None:
    #         return
    #     if isinstance(optimizer_config, dict):
    #         optimizer_config.setdefault('type', 'NoneOptimizerHook')
    #         hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
    #         print("GO INTO IF")
    #     else:
    #         optimizer_config = dict(lr=0.0, paramwise_options={'predictor': dict(fix_lr=True)})
    #         optimizer_config.setdefault('type', 'NoneOptimizerHook')
    #         hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
    #         print("GO INTO ELSE")
    #     self.register_hook(hook, priority='ABOVE_NORMAL')