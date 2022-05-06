import torch
import torchvision
import torchmetrics
import transformers
import pytorch_lightning as pl

from callbacks import Logger, BestModelMonitor

transformers.logging.set_verbosity_error()


class MnistVit(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.backbone = transformers.ViTModel.from_pretrained("PTM/vit-tiny-patch16-224")
        self.ff = torch.nn.Linear(192, 10)
        self.cls = torch.nn.Softmax(dim=-1)
        self.criterion = torch.nn.NLLLoss()
        self.train_metric = torchmetrics.Accuracy()
        self.valid_metric = torchmetrics.Accuracy()
        self.eps = 1e-7

    def forward(self, x):
        return self.cls(self.ff(self.backbone(x)[0][:, 0]))

    def training_step(self, batch, batch_idx):
        X, y = batch
        pred = self.forward(X)
        loss = self.criterion(torch.log(pred + self.eps), y)
        acc = self.train_metric(pred, y)

        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.log_dict
        # Set the log_dict to be synchronized in distributed training
        self.log_dict({'train_loss': loss, 'train_acc': acc}, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred = self.forward(X)
        loss = self.criterion(torch.log(pred + self.eps), y)
        acc = self.valid_metric(pred, y)

        # https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu.html#synchronize-validation-and-test-logging
        # Every thing is synchronized by default.
        self.log_dict({'valid_loss': loss, 'valid_acc': acc})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def train_dataloader(self):
        training_data = torchvision.datasets.FashionMNIST(root="data", train=True, download=True, transform=self.transform())
        return torch.utils.data.DataLoader(training_data, batch_size=256, shuffle=True, num_workers=16)

    def val_dataloader(self):
        val_data = torchvision.datasets.FashionMNIST(root="data", train=False, download=True, transform=self.transform())
        return torch.utils.data.DataLoader(val_data, batch_size=256, num_workers=16)

    def transform(self):
        feature_extractor = transformers.ViTFeatureExtractor.from_pretrained("PTM/vit-tiny-patch16-224")
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            torchvision.transforms.Lambda(lambda x: feature_extractor(x, return_tensors="pt")["pixel_values"][0])
        ])
        return transform

    def logger(self):
        return pl.loggers.TensorBoardLogger(save_dir="logs/")

    def callbacks(self):
        return [Logger(), BestModelMonitor(monitor="valid_acc", mode="max"), pl.callbacks.Timer()]
