import pytorch_lightning as pl
import numpy as np
import torch
import torchmetrics
import transformers
from pytorch_lightning.utilities import _TORCH_GREATER_EQUAL_1_10
from typing import Tuple

transformers.logging.set_verbosity_error()


class Clip(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        self.params = params

        self.image_backbone = transformers.AutoModel.from_pretrained(self.params.image_backbone_path)
        self.text_backbone = transformers.AutoModel.from_pretrained(self.params.text_backbone_path)
        

        # freeze params in backbone
        for index, (name, param) in enumerate(list(self.text_backbone.named_parameters())):
            if index<165:
                param.requires_grad = False
        #         print(index, "Freezed", name)
        #     else:
        #         print(index, "Unfreezed", name)
        for index, (name, param) in enumerate(list(self.image_backbone.named_parameters())):
            if index<164:
                param.requires_grad = False
        #         print(index, "Freezed", name)
        #     else:
        #         print(index, "Unfreezed", name)
        

        self.image_ffn = torch.nn.Linear(self.image_backbone.config.hidden_size, self.params.hidden_size, bias=False)

        self.text_ffn = torch.nn.Linear(self.text_backbone.config.hidden_size, self.params.hidden_size, bias=False)
    
        # appending *(1/0.07) might prevent the pytorch module to identify that as a learnable parameter
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # metrics
        self.train_clip_acc = torchmetrics.Accuracy()
        self.valid_clip_acc = torchmetrics.Accuracy()
        self.test_clip_acc = torchmetrics.Accuracy()

        self.train_clip_f1 = torchmetrics.F1Score(average='macro', num_classes=2)
        self.valid_clip_f1 = torchmetrics.F1Score(average='macro', num_classes=2)
        self.test_clip_f1 = torchmetrics.F1Score(average='macro', num_classes=2)

        self.train_clip_precision = torchmetrics.Precision(average='macro', num_classes=2)
        self.valid_clip_precision = torchmetrics.Precision(average='macro', num_classes=2)
        self.test_clip_precision = torchmetrics.Precision(average='macro', num_classes=2)

        self.train_clip_recall = torchmetrics.Recall(average='macro', num_classes=2)
        self.valid_clip_recall = torchmetrics.Recall(average='macro', num_classes=2)
        self.test_clip_recall = torchmetrics.Recall(average='macro', num_classes=2)


        # for onnx/torchscript export
        self.example_input_array = (
            torch.randint(high=10, size=(1, 3, self.params.image_size, self.params.image_size), dtype=torch.float),
            torch.randint(high=1000, size=(1, self.params.max_text_length))
        )

        # https://github.com/PyTorchLightning/pytorch-lightning/issues/3981
        # https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html?highlight=hyperparameters#lightningmodule-hyperparameters
        self.save_hyperparameters()


    def forward(self, item_image, item_name):
        if type(item_image) ==torch.Tensor:
            image_logist = self.image_ffn(self.image_backbone(item_image).last_hidden_state[:,0])
            text_logist = self.text_ffn(self.text_backbone(item_name).last_hidden_state[:,0])
        else:
            image_logist = self.image_ffn(self.image_backbone(**item_image).last_hidden_state[:,0])
            text_logist = self.text_ffn(self.text_backbone(**item_name).last_hidden_state[:,0])

        # normalized features
        image_logist = image_logist / torch.norm(image_logist, p=2, dim=1, keepdim=True)
        text_logist = text_logist / torch.norm(text_logist, p=2, dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = torch.matmul(image_logist, text_logist.transpose(0,1)) * logit_scale
        # logits_per_image = torch.matmul(image_logist, text_logist.transpose(0,1))

        logits_per_text = logits_per_image.transpose(0,1)

        return logits_per_image, logits_per_text, image_logist, text_logist

    def training_step(self, batch, batch_idx):
        
        item_image = batch['item_image']
        item_name = batch['item_name']

        logits_per_image, logits_per_text = self.forward(item_image, item_name)
        logits_per_image_max = logits_per_image.argmax(axis=1)
        logits_per_text_max = logits_per_text.argmax(axis=1)
        
        image_label_idx = torch.arange(len(logits_per_image), device=logits_per_image.device)
        flatten_image_label = torch.eye(len(logits_per_image), device=logits_per_image.device, dtype=torch.long).flatten()

        clip_cluster_max_acc = torch.sum(logits_per_image_max==image_label_idx)/len(logits_per_image)
        clip_item_max_acc = torch.sum(logits_per_text_max==image_label_idx)/len(logits_per_image)

        clip_loss = self.clip_loss(logits_per_image)

        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.log_dict
        # Set the log_dict to be synchronized in distributed training

        _log_dict = {
            'mean_logits': logits_per_image.mean(),
            'max_logits': logits_per_image.max(),
            'train_loss': clip_loss,
            'train_clip_cluster_max_acc':clip_cluster_max_acc,
            'train_clip_item_max_acc':clip_item_max_acc,
        }


        self.log_dict(_log_dict, sync_dist=True)

        return clip_loss

    def validation_step(self, batch, batch_idx):
        item_image = batch['item_image']
        item_name = batch['item_name']

        logits_per_image, logits_per_text = self.forward(item_image, item_name)

        logits_per_image_max = logits_per_image.argmax(axis=1)
        logits_per_text_max = logits_per_text.argmax(axis=1)

        image_label_idx = torch.arange(len(logits_per_image), device=logits_per_image.device)
        flatten_image_label = torch.eye(len(logits_per_image), device=logits_per_image.device, dtype=torch.long).flatten()

        clip_cluster_max_acc = torch.sum(logits_per_image_max==image_label_idx)/len(logits_per_image)
        clip_item_max_acc = torch.sum(logits_per_text_max==image_label_idx)/len(logits_per_image)

        clip_loss = self.clip_loss(logits_per_image)

        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.log_dict
        # Set the log_dict to be synchronized in distributed training
        _log_dict = {
            'valid_loss': clip_loss,
            'valid_clip_cluster_max_acc':clip_cluster_max_acc,
            'valid_clip_item_max_acc':clip_item_max_acc,
        }
        self.log_dict(_log_dict, sync_dist=True)

    def test_step(self, batch, batch_idx):
        item_image = batch['item_image']
        item_name = batch['item_name']

        logits_per_image, logits_per_text = self.forward(item_image, item_name)

        logits_per_image_max = logits_per_image.argmax(axis=1)
        logits_per_text_max = logits_per_text.argmax(axis=1)

        image_label_idx = torch.arange(len(logits_per_image), device=logits_per_image.device)
        flatten_image_label = torch.eye(len(logits_per_image), device=logits_per_image.device, dtype=torch.long).flatten()

        clip_cluster_max_acc = torch.sum(logits_per_image_max==image_label_idx)/len(logits_per_image)
        clip_item_max_acc = torch.sum(logits_per_text_max==image_label_idx)/len(logits_per_image)

        clip_loss = self.clip_loss(logits_per_image)

        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.log_dict
        # Set the log_dict to be synchronized in distributed training
        self.log_dict({
            'test_loss': clip_loss,
            'test_clip_cluster_max_acc':clip_cluster_max_acc,
            'test_clip_item_max_acc':clip_item_max_acc,

        }, sync_dist=True)

    def configure_optimizers(self):
        # optimizer = transformers.optimization.AdamW(self.parameters(), lr=self.params.lr, betas=(0.95, 0.999))
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.params.lr)
        optimizer = transformers.AdamW(self.parameters(), lr=self.params.lr)
        return optimizer


    # contrastive loss function, adapted from
    # https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        logits_flatten = logits.flatten()
        return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        cluster_loss = self.contrastive_loss(similarity)
        item_loss = self.contrastive_loss(similarity.transpose(0,1))
        return (item_loss + cluster_loss) / 2.0

    def resume_from_last(self):
        if self.params.resume:
            import os
            ckpt_dir = os.path.abspath(self.params.ckpt_dirpath)
            last_ckpt = [fn for fn in os.listdir(ckpt_dir) if fn.endswith(".ckpt")][-1]
            last_ckpt = f'{ckpt_dir}/{last_ckpt}'
            self.load_from_checkpoint(last_ckpt)
            print(f'resume from last ckpt {last_ckpt}')
            
    def get_image_feature(self, item_image):
        if type(item_image) ==torch.Tensor:
            image_logist = self.image_ffn(self.image_backbone(item_image).last_hidden_state[:,0])
        else:
            image_logist = self.image_ffn(self.image_backbone(**item_image).last_hidden_state[:,0])

        # normalized features
        image_logist = image_logist / torch.norm(image_logist, p=2, dim=1, keepdim=True)

        return image_logist

    def get_text_feature(self, input_ids,attention_mask=None):
        if type(input_ids) ==torch.Tensor:
            text_logist = self.text_ffn(self.text_backbone(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state[:,0])
        else:
            text_logist = self.text_ffn(self.text_backbone(**input_ids).last_hidden_state[:,0])

        # normalized features
        text_logist = text_logist / torch.norm(text_logist, p=2, dim=1, keepdim=True)

        return text_logist

    @torch.no_grad()
    def to_onnx(self, file_path, input_sample=None, **kwargs):

        mode = self.training

        if input_sample is None:
            if self.example_input_array is None:
                raise ValueError(
                    "Could not export to ONNX since neither `input_sample` nor"
                    " `model.example_input_array` attribute is set."
                )
            input_sample = self.example_input_array

        input_sample = self._apply_batch_transfer_handler(input_sample)
        item_image, input_ids = input_sample
        attention_mask = torch.randint(high=2, size=(1, self.params.max_text_length))
        # export cluster_backbone
        original_forward = self.forward
        self.forward = self.get_image_feature

        if not _TORCH_GREATER_EQUAL_1_10 and "example_outputs" not in kwargs:
            self.eval()
            if isinstance(item_image, Tuple):
                kwargs["example_outputs"] = self(*item_image)
            else:
                kwargs["example_outputs"] = self(item_image)

        kwargs['input_names'] = ["image_input"]
        kwargs['output_names'] = ["image_embeds"]                    
        torch.onnx.export(self, item_image, file_path+'/image.onnx', **kwargs)

        # export item_backbone
        self.forward = self.get_text_feature

        if not _TORCH_GREATER_EQUAL_1_10:
            self.eval()
            if isinstance(input_ids, Tuple):
                kwargs["example_outputs"] = self(*input_ids)
            else:
                kwargs["example_outputs"] = self(input_ids)

        kwargs['input_names'] = ["text_input",'text_attention_mask']  
        kwargs['output_names'] = ["text_embeds"]
              
        torch.onnx.export(self,(input_ids,attention_mask), file_path+'/text.onnx', **kwargs)

        # reset model state
        self.forward = original_forward
        self.train(mode)
