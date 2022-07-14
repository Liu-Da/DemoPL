import pytorch_lightning as pl
import torch
import torchvision
import transformers


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.save_hyperparameters(logger=False)

    def setup(self, stage):
        self.trainset = torchvision.datasets.FashionMNIST(root=self.params.data_path, train=True, download=True, transform=self.transform())
        self.validset = torchvision.datasets.FashionMNIST(root=self.params.data_path, train=False, download=True, transform=self.transform())
        self.testset = self.validset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers)


    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validset, batch_size=self.params.batch_size, num_workers=self.params.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.testset, batch_size=self.params.batch_size, num_workers=self.params.num_workers)

    def transform(self):
        feature_extractor = transformers.ViTFeatureExtractor.from_pretrained(self.params.ptm_path)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            torchvision.transforms.Lambda(lambda x: feature_extractor(x, return_tensors="pt")["pixel_values"][0])
        ])
        return transform

