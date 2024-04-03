# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

from mmengine.runner import IterBasedTrainLoop
from torch.utils.data import DataLoader

# add advane dataloader visualization
import logging
from tqdm import tqdm
from mmengine.logging import print_log


class TrainLoop(IterBasedTrainLoop):

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 max_iters: Optional[int] = None,
                 max_epochs: Union[int, float] = None,
                 **kwargs) -> None:

        if max_iters is None and max_epochs is None:
            raise RuntimeError('Please specify the `max_iters` or '
                               '`max_epochs` in `train_cfg`.')
        elif max_iters is not None and max_epochs is not None:
            raise RuntimeError('Only one of `max_iters` or `max_epochs` can '
                               'exist in `train_cfg`.')
        else:
            if max_iters is not None:
                iters = int(max_iters)
                assert iters == max_iters, ('`max_iters` should be a integer '
                                            f'number, but get {max_iters}')
            elif max_epochs is not None:
                if isinstance(dataloader, dict):
                    diff_rank_seed = runner._randomness_cfg.get(
                        'diff_rank_seed', False)
                    dataloader = runner.build_dataloader(
                        dataloader,
                        seed=runner.seed,
                        diff_rank_seed=diff_rank_seed)
                iters = max_epochs * len(dataloader)
            else:
                raise NotImplementedError
        super().__init__(
            runner=runner, dataloader=dataloader, max_iters=iters, **kwargs)

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        if self._iter > 0:
            print_log(
                f'Advance dataloader {self._iter} steps to skip data '
                'that has already been trained',
                logger='current',
                level=logging.WARNING)
            # visualiza the data load process
            for _ in tqdm(range(self._iter)):
                next(self.dataloader_iterator)
            s = input('Please type something to continue:')
        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and self._iter % self.val_interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model
