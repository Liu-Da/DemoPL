import pytorch_lightning as pl
import numpy as np
import torch
import torchmetrics
import transformers

transformers.logging.set_verbosity_error()


class ClipMatch(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        self.params = params
        
        self.backbone = transformers.AutoModel.from_pretrained(self.params.backbone_path)

        # # freeze params in backbone
        # for index, (name, param) in enumerate(list(self.backbone.named_parameters())):
        #     param.requires_grad = False
        #     print(index, "Freezed", name)

        self.cluster_name_ffn = torch.nn.Sequential(
            torch.nn.Linear(self.backbone.config.hidden_size, self.params.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(self.params.drop_rate),
            torch.nn.Linear(self.params.hidden_size, self.params.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(self.params.drop_rate),
            torch.nn.Linear(self.params.hidden_size, self.params.hidden_size),
        )

        self.item_name_ffn = torch.nn.Sequential(
            torch.nn.Linear(self.backbone.config.hidden_size, self.params.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(self.params.drop_rate),
            torch.nn.Linear(self.params.hidden_size, self.params.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(self.params.drop_rate),
            torch.nn.Linear(self.params.hidden_size, self.params.hidden_size),
        )
        # appending *(1/0.07) might prevent the pytorch module to identify that as a learnable parameter
        self.logit_scale = torch.nn.Parameter(torch.ones([]) *np.log(1 / 0.07))

        # metrics
        self.train_clip_acc = torchmetrics.Accuracy()
        self.valid_clip_acc = torchmetrics.Accuracy()
        self.test_clip_acc = torchmetrics.Accuracy()

        self.train_clip_f1 = torchmetrics.F1Score()
        self.valid_clip_f1 = torchmetrics.F1Score()
        self.test_clip_f1 = torchmetrics.F1Score()

        self.train_clip_precision = torchmetrics.Precision()
        self.valid_clip_precision = torchmetrics.Precision()
        self.test_clip_precision = torchmetrics.Precision()

        self.train_clip_recall = torchmetrics.Recall()
        self.valid_clip_recall = torchmetrics.Recall()
        self.test_clip_recall = torchmetrics.Recall()

        # self.train_clip_pr_curve = torchmetrics.PrecisionRecallCurve()
        # self.valid_clip_pr_curve = torchmetrics.PrecisionRecallCurve()
        # self.test_clip_pr_curve = torchmetrics.PrecisionRecallCurve()

        # for onnx/torchscript export
        self.example_input_array = (
            torch.randint(high=1000, size=(1, self.params.max_cluster_name_length)),
            torch.randint(high=1000, size=(1, self.params.max_item_name_length))
        )

        # https://github.com/PyTorchLightning/pytorch-lightning/issues/3981
        # https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html?highlight=hyperparameters#lightningmodule-hyperparameters
        self.save_hyperparameters()


    def forward(self, cluster_name, item_name):

        cluster_name_logist = self.cluster_name_ffn(self.backbone(cluster_name).last_hidden_state[:,0])
        item_name_logist = self.item_name_ffn(self.backbone(item_name).last_hidden_state[:,0])

        # normalized features
        cluster_name_logist = cluster_name_logist / cluster_name_logist.norm(p=2, dim=-1, keepdim=True)
        item_name_logist = item_name_logist / item_name_logist.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_cluster = torch.matmul(cluster_name_logist, item_name_logist.transpose(0,1)) * logit_scale
        logits_per_item = logits_per_cluster.transpose(0,1)

        return logits_per_cluster, logits_per_item

    def training_step(self, batch, batch_idx):
        cluster_name = batch['cluster_name']
        item_name = batch['item_name']

        logits_per_cluster, logits_per_item = self.forward(cluster_name, item_name)

        logits_per_cluster_max = logits_per_cluster.argmax(axis=1)
        logits_per_item_max = logits_per_cluster.argmax(axis=0)
        
        cluster_label_idx = torch.arange(len(logits_per_cluster), device=logits_per_cluster.device)

        clip_cluster_max_acc = torch.sum(logits_per_cluster_max==cluster_label_idx)/len(logits_per_cluster)
        clip_item_max_acc = torch.sum(logits_per_item_max==cluster_label_idx)/len(logits_per_cluster)

        clip_loss = self.clip_loss(logits_per_cluster)

        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.log_dict
        # Set the log_dict to be synchronized in distributed training

        _log_dict = {
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
        logits_per_item_max = logits_per_cluster.argmax(axis=0)

        cluster_label_idx = torch.arange(len(logits_per_cluster), device=logits_per_cluster.device)

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
        
        flatten_cluster_res_5 = (logits_per_cluster>0.5*self.logit_scale.exp()).float().flatten()
        flatten_cluster_label = torch.eye(len(logits_per_cluster), device=logits_per_cluster.device, dtype=torch.long).flatten()

        clip_acc = self.test_clip_acc(flatten_cluster_res_5,flatten_cluster_label)
        # precision, recall, thresholds = self.test_clip_pr_curve(flatten_cluster_res,flatten_cluster_label)

        clip_loss = self.clip_loss(logits_per_cluster)

        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.log_dict
        # Set the log_dict to be synchronized in distributed training
        self.log_dict({
            'test_loss': clip_loss,
            'test_acc': clip_acc,

        }, sync_dist=True)

    def configure_optimizers(self):
        optimizer = transformers.optimization.AdamW(self.parameters(), lr=self.params.lr, betas=(0.95, 0.999))

        n_batches, epochs = 100, 10  # int(np.ceil(len(self.train_dataloader) / self.config.batch_size))
        total_steps = epochs * n_batches
        n_warmup_steps = int(total_steps * 0.01)
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, n_warmup_steps, total_steps)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    # contrastive loss function, adapted from
    # https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        logits_flatten = logits.flatten()
        return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        cluster_loss = self.contrastive_loss(similarity)
        item_loss = self.contrastive_loss(similarity.transpose(0,1))
        return (item_loss + cluster_loss) / 2.0