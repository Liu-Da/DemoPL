import unicodedata
import pytorch_lightning as pl


class Logger(pl.Callback):

    def __init__(self):
        super().__init__()

    @staticmethod
    def print_table(logs):
        metrics = list(logs.keys())
        val = list(logs.values())
        val = [i.detach().cpu().item() for i in val]

        assert len(metrics) > 0 and len(metrics) == len(val)

        def wide_chars(s):
            return sum(unicodedata.east_asian_width(x) == 'W' for x in s)

        def width(s):
            return len(s) + wide_chars(s)

        _max = max([width(str(i)) for i in metrics + val])

        for i, j in zip(metrics, val):
            print("-" * (_max * 2 + 5))
            print(f" {i:^{_max-wide_chars(str(i))}} | {str(j):^{_max-wide_chars(str(j))}} ")
        print("-" * (_max * 2 + 5))

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.is_global_zero:
            logs = trainer.callback_metrics
            current_epoch = trainer.current_epoch + 1
            print(f"\nEpoch : {current_epoch}")
            self.print_table(logs)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.is_global_zero:
            logs = trainer.callback_metrics
            print("\nFinal Results")
            self.print_table(logs)
