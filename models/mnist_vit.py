import pytorch_lightning as pl
import torch
import torchmetrics
import transformers

transformers.logging.set_verbosity_error()


class MnistVit(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        self.params = params
        self.backbone = transformers.ViTModel.from_pretrained(self.params.ptm_path)
        self.ff0 = torch.nn.Linear(self.backbone.config.hidden_size, self.params.hidden_size)
        self.dp = torch.nn.Dropout(p=self.params.drop_rate)
        self.ff1 = torch.nn.Linear(self.params.hidden_size, self.params.num_classes)
        self.cls = torch.nn.Softmax(dim=-1)
        self.criterion = torch.nn.NLLLoss()
        self.train_metric = torchmetrics.Accuracy()
        self.valid_metric = torchmetrics.Accuracy()
        self.eps = 1e-8

        # for onnx/torchscript export
        self.example_input_array = torch.randn(1, 3, 224, 224)

        # https://github.com/PyTorchLightning/pytorch-lightning/issues/3981
        # https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html?highlight=hyperparameters#lightningmodule-hyperparameters
        self.save_hyperparameters()

        # TODO:
        # * set @pl.utilities.rank_zero_only
        for index, (name, param) in enumerate(list(self.named_parameters())):
            if index < 100:
                param.requires_grad = False
            #     print(index, "Freezed", name)
            # else:
            #     print(index, "Trainable", name)

    def forward(self, x):
        return self.cls(self.ff1(self.dp(self.ff0(self.backbone(x)[0][:, 0]))))

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

    def test_step(self, batch, batch_idx):
        X, y = batch
        pred = self.forward(X)
        loss = self.criterion(torch.log(pred + self.eps), y)
        acc = self.valid_metric(pred, y)

        # https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu.html#synchronize-validation-and-test-logging
        # Every thing is synchronized by default.
        self.log_dict({'test_loss': loss, 'test_acc': acc})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.lr)
        return optimizer
