import logging
import numpy as np
import pytorch_lightning as pl


class BestModelMonitor(pl.Callback):
    def __init__(self, monitor, mode="max"):
        super().__init__()
        self.monitor = monitor
        self.best = 0
        self.best_epoch = 0

        if mode == "max":
            self.monitor_op = np.greater
        else:
            self.monitor_op = np.less

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logs = trainer.callback_metrics
        current_epoch = trainer.current_epoch + 1
        current = self.get_monitor_value(logs)

        if self.monitor_op(current, self.best):
            self.best = current
            self.best_epoch = current_epoch
        if trainer.is_global_zero:
            print(f'Best {self.monitor} until now is {self.best} at epoch {self.best_epoch}')

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(f"BestModelMonitor conditioned on metric {self.monitor} which is not available.\n"
                            f"Available metrics are: {','.join(list(logs.keys()))}")
        return monitor_value.detach().cpu().item()