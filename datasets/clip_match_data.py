import pandas as pd
import pytorch_lightning as pl
import torch
import transformers


class ClipMatchDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.save_hyperparameters(logger=False)

    def setup(self, stage=None):
        self.dataset = ClipMatchDataset(self.params) 

        train_length = int(len(self.dataset) * 0.9)
        valid_length = len(self.dataset) - train_length


        self.trainset, self.validset = torch.utils.data.random_split(
            dataset=self.dataset,
            lengths=[train_length, valid_length],
            generator=torch.Generator().manual_seed(0)
        )
        self.testset = self.validset
        print(f"train: {len(self.trainset)}")
        print(f"valid: {len(self.validset)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset, 
            batch_size=self.params.batch_size, 
            num_workers=self.params.num_workers,
            collate_fn=self.dataset.collate_fn,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validset,
            batch_size=self.params.batch_size, 
            num_workers=self.params.num_workers,
            collate_fn=self.dataset.collate_fn
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.params.batch_size, 
            num_workers=self.params.num_workers,
            collate_fn=self.dataset.collate_fn
        )


class ClipMatchDataset(torch.utils.data.Dataset):
    def __init__(self, params):
        self.params = params
        self.df = self.read_pd().reset_index(drop=True)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(params.item_backbone_path)

    def read_pd(self):
        return pd.read_parquet(self.params.data_path) 


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        cluster_name, item_name = self.df.loc[idx,['cluster_name','item_name']]
        return cluster_name, item_name

    def collate_fn(self, batch):
        batchT = list(map(list, zip(*batch)))   
        batch_tensor = {}
        batch_tensor["cluster_name"] = self.tokenizer(batchT[0], padding='max_length', truncation=True, max_length=self.params.max_cluster_name_length, return_tensors="pt")
        batch_tensor["item_name"] = self.tokenizer(batchT[1], padding='max_length', truncation=True, max_length=self.params.max_item_name_length, return_tensors="pt")
        return batch_tensor


if __name__ == '__main__':
    import omegaconf
    params = omegaconf.OmegaConf.create()
    params.item_backbone_path = 'PTM/xlmr-ID-pretrained'
    params.data_path = "/data/xiaoqing.tang/02experiment/pytroch_lightning/x/data/clip_match.parquet"

    params.batch_size = 16
    params.num_workers = 0
    params.max_cluster_name_length = 16
    params.max_item_name_length = 64
    dm = ClipMatchDataModule(params)
    dm.setup()
    train_dataloader =dm.train_dataloader()
    for i in next(iter(train_dataloader)).values():
        print(i)
    print([i.input_ids.shape for i in next(iter(train_dataloader)).values()])
    # print([i.shape for i in next(iter(train_dataloader)).values()])