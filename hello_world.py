import pytorch_lightning as pl

from models import MnistVit
from utils import gpu_detecter

EPOCH = 50

gpu_num = gpu_detecter()

model = MnistVit()
train_dataloader = model.train_dataloader()
val_dataloader = model.val_dataloader()
tb_logger = model.logger()
callbacks = model.callbacks()

trainer = pl.Trainer(accelerator="gpu" if gpu_num > 1 else "cpu",
                     devices=gpu_num if gpu_num > 1 else None,
                     strategy="ddp" if gpu_num > 1 else None,
                     max_epochs=EPOCH,
                     logger=tb_logger,
                     callbacks=callbacks,
                     enable_progress_bar=False,
                     num_sanity_val_steps=0)

trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
