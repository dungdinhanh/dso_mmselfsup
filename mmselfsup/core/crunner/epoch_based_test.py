# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import platform
import shutil
import time
import warnings

import torch

import mmcv
from mmcv.runner.base_runner import BaseRunner
from mmcv.runner.epoch_based_runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.utils import get_host_info


from mmcv.runner.epoch_based_runner import EpochBasedRunner


@RUNNERS.register_module()
class EpochBasedRunnerLogMin(EpochBasedRunner):
    """KD-based Runner.

    This runner train models epoch by epoch. For each epoch, the runner feed in the teacher model.
    """
    def __init__(self,
                 model,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None):
        super(EpochBasedRunnerLogMin, self).__init__(
                model=model,
                batch_processor=batch_processor,
                optimizer=optimizer,
                work_dir=work_dir,
                logger=logger,
                meta=meta,
                max_iters=max_iters,
                max_epochs=max_epochs)

        self.min_loss_epoch = []
        self.MAX_VAL = 99999
        self.current_min_loss_iter1 = self.MAX_VAL
        self.current_min_loss_iter2 = self.MAX_VAL
        self.current_min_loss_epoch1 = self.MAX_VAL
        self.current_min_loss_epoch2 = self.MAX_VAL
        self.log_min_epoch_file = os.path.join(self.work_dir, "min_loss_epochs.csv")
        self.f = None

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs, save_min = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
            if self.current_min_loss_iter1 > save_min[0]:
                self.current_min_loss_iter1 = save_min[0]
            if self.current_min_loss_iter2 > save_min[1]:
                self.current_min_loss_iter2 = save_min[1]
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
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1
        self.current_min_loss_epoch1 = self.current_min_loss_iter1
        self.current_min_loss_epoch2 = self.current_min_loss_iter2
        self.min_loss_epoch.append([self.current_min_loss_epoch1, self.current_min_loss_iter2])
        self.f.write("%d, %f, %f\n"%(self._epoch, self.current_min_loss_epoch1, self.current_min_loss_epoch2))
        self.current_min_loss_epoch1 = self.MAX_VAL
        self.current_min_loss_epoch2 = self.MAX_VAL
        self.current_min_loss_iter1 = self.MAX_VAL
        self.current_min_loss_iter2 = self.MAX_VAL
        self.call_hook('after_train_epoch')
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
        self.f = open(self.log_min_epoch_file, "w")
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
        self.f.close()

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
