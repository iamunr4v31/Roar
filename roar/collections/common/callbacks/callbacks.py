import time

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only


class LogEpochTimeCallback(Callback):
    """Simple callback that logs how long each epoch takes, in seconds, to a pytorch lightning log"""

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        curr_time = time.time()
        duration = curr_time - self.epoch_start
        trainer.logger.log_metrics({"epoch_time": duration}, step=trainer.global_step)
