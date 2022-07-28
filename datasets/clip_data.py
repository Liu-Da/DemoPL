from matplotlib import image
import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import requests
import io
import urllib.request as request


class ClipDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.save_hyperparameters(logger=False)

    def setup(self, stage=None):
        self.dataset = ClipDataset(self.params) 

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
        print(f"test: {len(self.testset)}")

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


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, params):
        self.params = params
        self.df = self.read_pd()
        self.feature_extractor = transformers.ViTFeatureExtractor.from_pretrained(params.image_backbone_path)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(params.text_backbone_path)

    def read_pd(self):
        return pd.read_parquet(self.params.data_path).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path, item_name = self.df.loc[idx,['image_path', 'name']]
        try:
            image_full_path = '/fangzheng_bigdata/fzdata/img_search/images/0817_48mil_items_7sites/'+image_path
            item_image = Image.open(image_full_path)
            if len(item_image.size) == 2 or item_image.mode != 'RGB':
                item_image = item_image.convert('RGB')
            return item_image, item_name
        except:
            r = requests.get(image_url, stream=True)
            item_image = Image.open(io.BytesIO(r.content))
            if len(item_image.size) == 2 or item_image.mode != 'RGB':
                item_image = item_image.convert('RGB')
            return item_image, item_name
        # except Exception as e:
            # print(str(e))
            # print(image_path)

    def collate_fn(self, batch):
        batch = filter(lambda x : x is not None, batch)
        batchT = list(map(list, zip(*batch)))
        batch_tensor = {}
        batch_tensor["item_image"] = self.feature_extractor(batchT[0],size=self.params.image_size,return_tensors="pt")
        batch_tensor["item_name"] = self.tokenizer(batchT[1], padding='max_length', truncation=True, max_length=self.params.max_text_length, return_tensors="pt")
        return batch_tensor


if __name__ == '__main__':
    import omegaconf
    params = omegaconf.OmegaConf.create()
    params.image_backbone_path = 'PTM/vit-base-patch16-224'
    params.text_backbone_path = 'PTM/xlmr-ID-pretrained'
    params.data_path = "/data/xiaoqing.tang/02experiment/pytroch_lightning/x/data/clip.parquet"

    params.batch_size = 2
    params.num_workers = 0
    params.image_size = 384
    params.max_text_length = 64
    dm = ClipDataModule(params)
    dm.setup()
    train_dataloader =dm.train_dataloader()
    # for i, batch in enumerate(train_dataloader):
    #     print(batch["item_image"].pixel_values.size())
    #     print(batch["item_name"].input_ids.shape)
    

    img,text= next(iter(train_dataloader)).values()
    print(img.keys())
    print(text.keys())
    print(img.pixel_values.size())
    print(text.input_ids.size())
    # print([i for i in next(iter(train_dataloader)).values()])