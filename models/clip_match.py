import pytorch_lightning as pl
import numpy as np
import torch
import torchmetrics
import transformers
from pytorch_lightning.utilities import _TORCH_GREATER_EQUAL_1_10
from typing import List, Dict, Tuple


transformers.logging.set_verbosity_error()


class ClipMatch(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        self.params = params
        
        self.cluster_backbone = transformers.AutoModel.from_pretrained(self.params.cluster_backbone_path)
        self.item_backbone = transformers.AutoModel.from_pretrained(self.params.item_backbone_path)

        # # freeze params in backbone
        # for index, (name, param) in enumerate(list(self.cluster_backbone.named_parameters())):
        #     if index<53:
        #         param.requires_grad = False
        #         print(index, "Freezed", name)
        # for index, (name, param) in enumerate(list(self.item_backbone.named_parameters())):
        #     if index<53:
        #         param.requires_grad = False
        #         print(index, "Freezed", name)


        self.cluster_name_ffn = torch.nn.Sequential(
            torch.nn.Linear(self.cluster_backbone.config.hidden_size, self.params.hidden_size)
        )

        self.item_name_ffn = torch.nn.Sequential(
            torch.nn.Linear(self.item_backbone.config.hidden_size, self.params.hidden_size)
        )
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
            torch.randint(high=1000, size=(1, self.params.max_cluster_name_length)),
            torch.randint(high=1000, size=(1, self.params.max_item_name_length)),
        )

        # https://github.com/PyTorchLightning/pytorch-lightning/issues/3981
        # https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html?highlight=hyperparameters#lightningmodule-hyperparameters
        self.save_hyperparameters()


    def forward(self, cluster_name, item_name):
        if type(cluster_name) ==torch.Tensor:
            cluster_name_logist = self.cluster_name_ffn(self.cluster_backbone(cluster_name).last_hidden_state[:,0])
            item_name_logist = self.item_name_ffn(self.item_backbone(item_name).last_hidden_state[:,0])
        else:
            cluster_name_logist = self.cluster_name_ffn(self.cluster_backbone(**cluster_name).last_hidden_state[:,0])
            item_name_logist = self.item_name_ffn(self.item_backbone(**item_name).last_hidden_state[:,0])

        # normalized features
        cluster_name_logist = cluster_name_logist / cluster_name_logist.norm(p=2, dim=-1, keepdim=True)
        item_name_logist = item_name_logist / item_name_logist.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_cluster = torch.matmul(cluster_name_logist, item_name_logist.transpose(0,1)) * logit_scale

        logits_per_item = logits_per_cluster.transpose(0,1)

        return logits_per_cluster, logits_per_item, cluster_name_logist,item_name_logist

    def training_step(self, batch, batch_idx):
        cluster_name = batch['cluster_name']
        item_name = batch['item_name']

        logits_per_cluster, logits_per_item = self.forward(cluster_name, item_name)
        _logits_per_cluster, _logits_per_item = self.forward(cluster_name, item_name)

        logits_per_cluster_max = logits_per_cluster.argmax(axis=1)
        logits_per_item_max = logits_per_item.argmax(axis=1)
        
        cluster_label_idx = torch.arange(len(logits_per_cluster), device=logits_per_cluster.device)
        flatten_cluster_label = torch.eye(len(logits_per_cluster), device=logits_per_cluster.device, dtype=torch.long).flatten()

        clip_cluster_max_acc = torch.sum(logits_per_cluster_max==cluster_label_idx)/len(logits_per_cluster)
        clip_item_max_acc = torch.sum(logits_per_item_max==cluster_label_idx)/len(logits_per_cluster)

        clip_loss = self.clip_loss(logits_per_cluster)
        clip_loss += self.clip_loss(_logits_per_cluster)

        
        clip_loss += self.compute_kl_loss(logits_per_cluster, _logits_per_cluster) * 0.5

        diff = torch.sum(logits_per_cluster-_logits_per_cluster)
        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.log_dict
        # Set the log_dict to be synchronized in distributed training

        _log_dict = {
            'diff': diff,
            'mean_logits': logits_per_cluster.mean(),
            'max_logits': logits_per_cluster.max(),
            # 'logit_scale': logit_scale,
            'train_loss': clip_loss,
            'train_clip_cluster_max_acc':clip_cluster_max_acc,
            'train_clip_item_max_acc':clip_item_max_acc,
        }

        self.log_dict(_log_dict, sync_dist=True)

        return clip_loss

    def validation_step(self, batch, batch_idx):
        cluster_name = batch['cluster_name']
        item_name = batch['item_name']

        logits_per_cluster, logits_per_item = self.forward(cluster_name, item_name)

        logits_per_cluster_max = logits_per_cluster.argmax(axis=1)
        logits_per_item_max = logits_per_item.argmax(axis=1)

        cluster_label_idx = torch.arange(len(logits_per_cluster), device=logits_per_cluster.device)
        flatten_cluster_label = torch.eye(len(logits_per_cluster), device=logits_per_cluster.device, dtype=torch.long).flatten()

        clip_cluster_max_acc = torch.sum(logits_per_cluster_max==cluster_label_idx)/len(logits_per_cluster)
        clip_item_max_acc = torch.sum(logits_per_item_max==cluster_label_idx)/len(logits_per_cluster)

        clip_loss = self.clip_loss(logits_per_cluster)

        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.log_dict
        # Set the log_dict to be synchronized in distributed training
        _log_dict = {
            'valid_loss': clip_loss,
            'valid_clip_cluster_max_acc':clip_cluster_max_acc,
            'valid_clip_item_max_acc':clip_item_max_acc,
        }


        self.log_dict(_log_dict, sync_dist=True)

    def test_step(self, batch, batch_idx):
        cluster_name = batch['cluster_name']
        item_name = batch['item_name']

        logits_per_cluster, logits_per_item = self.forward(cluster_name, item_name)
        
        logits_per_cluster_max = logits_per_cluster.argmax(axis=1)
        logits_per_item_max = logits_per_item.argmax(axis=1)

        cluster_label_idx = torch.arange(len(logits_per_cluster), device=logits_per_cluster.device)
        flatten_cluster_label = torch.eye(len(logits_per_cluster), device=logits_per_cluster.device, dtype=torch.long).flatten()

        clip_cluster_max_acc = torch.sum(logits_per_cluster_max==cluster_label_idx)/len(logits_per_cluster)
        clip_item_max_acc = torch.sum(logits_per_item_max==cluster_label_idx)/len(logits_per_cluster)

        clip_loss = self.clip_loss(logits_per_cluster)

        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.log_dict
        # Set the log_dict to be synchronized in distributed training
        self.log_dict({
            'test_loss': clip_loss,
            'test_clip_cluster_max_acc':clip_cluster_max_acc,
            'test_clip_item_max_acc':clip_item_max_acc,
        }, sync_dist=True)

    def configure_optimizers(self):
        optimizer = transformers.optimization.AdamW(self.parameters(), lr=self.params.lr, betas=(0.95, 0.999))
        # optimizer = transformers.AdamW(self.parameters(), lr=self.params.lr)
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

    def compute_kl_loss(self, p, q, pad_mask=None):
        """
        Compute symmetric Kullback-Leibler divergence Loss.
        """
        p = p.flatten()
        q = q.flatten()
        p_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(p, dim=-1),
                        torch.nn.functional.softmax(q, dim=-1), reduction='none')
        q_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(q, dim=-1),
                        torch.nn.functional.softmax(p, dim=-1), reduction='none')

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    
    def get_cluster_feature(self, input_ids,attention_mask=None):
        if type(input_ids) ==torch.Tensor:
            cluster_name_logist = self.cluster_name_ffn(self.cluster_backbone(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state[:,0])
        else:
            cluster_name_logist = self.cluster_name_ffn(self.cluster_backbone(**input_ids).last_hidden_state[:,0])

        # normalized features
        cluster_name_logist = cluster_name_logist / cluster_name_logist.norm(p=2, dim=-1, keepdim=True)

        return cluster_name_logist

    def get_item_feature(self, input_ids,attention_mask=None):
        if type(input_ids) ==torch.Tensor:
            item_name_logist = self.item_name_ffn(self.item_backbone(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state[:,0])
        else:
            item_name_logist = self.item_name_ffn(self.item_backbone(**input_ids).last_hidden_state[:,0])

        # normalized features
        item_name_logist = item_name_logist / item_name_logist.norm(p=2, dim=-1, keepdim=True)

        return item_name_logist

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
        cluster_name, item_name = input_sample
        cluster_attention_mask = torch.randint(high=2, size=(1, self.params.max_cluster_name_length))
        item_attention_mask = torch.randint(high=2, size=(1, self.params.max_item_name_length))

        # export cluster_backbone
        original_forward = self.forward
        self.forward = self.get_cluster_feature

        if not _TORCH_GREATER_EQUAL_1_10 and "example_outputs" not in kwargs:
            self.eval()
            if isinstance(cluster_name, Tuple):
                kwargs["example_outputs"] = self(*cluster_name)
            else:
                kwargs["example_outputs"] = self(cluster_name)

        kwargs['input_names'] = ["cluster_input","cluster_attention_mask"]    
        kwargs['output_names'] = ["cluster_embeds"]                    
        torch.onnx.export(self, (cluster_name,cluster_attention_mask), file_path+'/cluster.onnx', **kwargs)

        # export item_backbone
        self.forward = self.get_item_feature

        if not _TORCH_GREATER_EQUAL_1_10:
            self.eval()
            if isinstance(item_name, Tuple):
                kwargs["example_outputs"] = self(*item_name)
            else:
                kwargs["example_outputs"] = self(item_name)

        kwargs['input_names'] = ["item_input","item_attention_mask"]    
        kwargs['output_names'] = ["item_embeds"]                    
        torch.onnx.export(self, (item_name,item_attention_mask), file_path+'/item.onnx', **kwargs)

        # reset model state
        self.forward = original_forward
        self.train(mode)