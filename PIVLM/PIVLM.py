import datetime
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dateutil import tz
from einops import rearrange
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
from datasets.datamodule import CLIP_DataModule
from datasets.datasets import CLIPDataset, CLIP_SSL_Dataset
from datasets.datasets import clip_ssl_collate_fn, clip_collate_fn
from datasets.transforms import DataTransforms
from backbones.encoder import BertEncoder, ImageEncoder
from torch import distributed as dist


class PIVLM(LightningModule):
    """Pytorch lightning implementation of PIVLM"""

    def __init__(self,
                 img_encoder: str = "vit_base",
                 freeze_bert: bool = False,
                 emb_dim: int = 128,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 # TODO: tune this hyperparameter
                 local_temperature: float = 0.1,
                 proto_temperature: float = 0.2,
                 num_prototypes: int = 512,
                 bidirectional: bool = True,
                 use_local_atten: bool = False,
                 num_heads: int = 1,
                 lamb: float = 0.75,
                 lambda_1: float = 1,
                 lambda_2: float = 1,
                 lambda_3: float = 1,
                 freeze_prototypes_epochs: int = 1,
                 sinkhorn_iterations: int = 3,
                 epsilon: float = 0.05,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        # init encoders
        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)
        self.text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert)

        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)
        # sentence local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)

        self.prototype_layer = nn.Linear(emb_dim, num_prototypes, bias=False)
        if self._use_ddp_or_dpp2(self.trainer):
            self.get_assignments = self.distributed_sinkhorn
        else:
            self.get_assignments = self.sinkhorn

    def forward(self, batch, batch_idx, split="train"):
        """Forward step of our method"""

        loss_ita, loss_local, loss_proto, acc1, acc5 = self.vl_forward(batch)

        loss_aug_global, loss_patch, loss_aug_proto = self.ssl_forward(batch)

        return (loss_ita, loss_local, loss_proto, loss_aug_global,
                loss_patch, loss_aug_proto, acc1, acc5)

    def text(self):
        """report_feat_q, word_feat_q, word_attn_q, sents = self.text_encoder_q(
            a, b, c)
        report_emb_q = self.text_encoder_q.global_embed(report_feat_q)
        report_emb_q = F.normalize(report_emb_q, dim=-1)
        return report_emb_q"""
        return self.text_encoder_q

    def image(self):
        # Forward of image encoder
        """img_feat_q, patch_feat_q = self.img_encoder_q(image)
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)
        return img_emb_q"""
        return self.img_encoder_q

    def vl_forward(self, batch):
        # Forward of query image encoder
        img_feat_q, patch_feat_q = self.img_encoder_q(
            batch["imgs"])
        patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        # Forward of query text encoder
        report_feat_q, word_feat_q, word_attn_q, sents = self.text_encoder_q(
            batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"])
        word_emb_q = self.text_encoder_q.local_embed(word_feat_q)
        word_emb_q = F.normalize(word_emb_q, dim=-1)
        report_emb_q = self.text_encoder_q.global_embed(report_feat_q)
        report_emb_q = F.normalize(report_emb_q, dim=-1)

        bz = img_emb_q.size(0)
        labels = torch.arange(bz).type_as(report_emb_q).long()

        scores = img_emb_q.mm(report_emb_q.t())
        scores /= self.hparams.softmax_temperature
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores1, labels)
        loss_ita = loss0 + loss1

        # compute retrieval accuracy
        i2t_acc1, i2t_acc5 = self.precision_at_k(
            scores, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            scores1, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        ########### Token-level alignment ################
        # cross attention patch to sentences
        mask = torch.from_numpy(np.array(sents)[:, 1:] == "[PAD]").type_as(
            batch["imgs"]).bool()

        if self.hparams.use_local_atten:
            word_atten_output, _ = self.word_local_atten_layer(
                word_emb_q, patch_emb_q, patch_emb_q)
        else:
            atten_sim = torch.bmm(word_emb_q, patch_emb_q.permute(0, 2, 1))
            atten_scores = F.softmax(
                atten_sim / self.hparams.local_temperature, dim=-1)  # bz, 196, 111
            word_atten_output = torch.bmm(atten_scores, patch_emb_q)

        word_atten_output = F.normalize(word_atten_output, dim=-1)

        with torch.no_grad():
            atten_weights = word_attn_q.detach()
            word_atten_weights = []
            for i in range(bz):
                atten_weight = atten_weights[i]
                nonzero = atten_weight.nonzero().squeeze()
                low = torch.quantile(atten_weight[nonzero], 0.1)
                high = torch.quantile(atten_weight[nonzero], 0.9)
                atten_weight[nonzero] = atten_weight[nonzero].clip(low, high)
                word_atten_weights.append(atten_weight.clone())
            word_atten_weights = torch.stack(word_atten_weights)

        word_atten_weights /= word_atten_weights.sum(dim=1, keepdims=True)

        word_sim = torch.bmm(word_emb_q, word_atten_output.permute(
            0, 2, 1)) / self.hparams.local_temperature
        word_num = word_sim.size(1)
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")
        targets = torch.arange(word_num).type_as(
            word_emb_q).long().repeat(bz)
        loss_word_1 = torch.sum(F.cross_entropy(
            word_sim_1, targets, reduction="none") * word_atten_weights.view(-1)) / bz

        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = torch.sum(F.cross_entropy(
            word_sim_2, targets, reduction="none") * word_atten_weights.view(-1)) / bz

        loss_word = (loss_word_1 + loss_word_2) / 2.

        if self.hparams.bidirectional:
            # Try not use atten layer
            if self.hparams.use_local_atten:
                patch_atten_output, _ = self.patch_local_atten_layer(
                    patch_emb_q, word_emb_q, word_emb_q, key_padding_mask=mask)
            else:
                atten_sim = torch.bmm(patch_emb_q, word_emb_q.permute(0, 2, 1))
                patch_num = patch_emb_q.size(1)
                atten_sim[mask.unsqueeze(1).repeat(
                    1, patch_num, 1)] = float("-inf")
                atten_scores = F.softmax(
                    atten_sim / self.hparams.local_temperature, dim=-1)  # bz, 196, 111
                patch_atten_output = torch.bmm(atten_scores, word_emb_q)

            # patch_atten_output: bz, 196, 128
            if "vit" not in self.hparams.img_encoder:
                patch_atten_output = F.normalize(patch_atten_output, dim=-1)
                patch_num = patch_atten_output.size(1)
                patch_atten_weights = torch.ones(
                    bz, patch_num).type_as(batch["imgs"]) / patch_num

            else:
                with torch.no_grad():
                    img_attn_map = self.img_encoder_q.model.blocks[-1].attn.attention_map.detach(
                    )
                    atten_weights = img_attn_map[:, :, 0, 1:].mean(dim=1)
                    patch_atten_weights = []
                    for i in range(bz):
                        atten_weight = atten_weights[i]
                        atten_weight = atten_weight.clip(torch.quantile(
                            atten_weight, 0.1), torch.quantile(atten_weight, 0.9))
                        patch_atten_weights.append(atten_weight.clone())
                    patch_atten_weights = torch.stack(patch_atten_weights)

                patch_atten_weights /= patch_atten_weights.sum(
                    dim=1, keepdims=True)

            patch_sim = torch.bmm(patch_emb_q, patch_atten_output.permute(
                0, 2, 1)) / self.hparams.local_temperature
            patch_num = patch_sim.size(1)
            patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
            targets = torch.arange(patch_num).type_as(
                patch_emb_q).long().repeat(bz)
            # loss_patch_1 = F.cross_entropy(patch_sim_1, targets)
            loss_patch_1 = torch.sum(F.cross_entropy(
                patch_sim_1, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
            loss_patch_2 = torch.sum(F.cross_entropy(
                patch_sim_2, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            loss_patch = (loss_patch_1 + loss_patch_2) / 2.

            loss_local = loss_patch + loss_word

        else:

            loss_local = loss_word

        # Sentence level
        word_emb_q_down = word_emb_q[:, ::2, :]
        patch_emb_q_down = patch_emb_q[:, ::2, :]

        if self.hparams.use_local_atten:
            word_atten_output_down, _ = self.word_local_atten_layer(
                word_emb_q_down, patch_emb_q_down, patch_emb_q_down)
        else:
            atten_sim_down = torch.bmm(word_emb_q_down, patch_emb_q_down.permute(0, 2, 1))
            atten_scores_down = F.softmax(
                atten_sim_down / self.hparams.local_temperature, dim=-1)
            word_atten_output_down = torch.bmm(atten_scores_down, patch_emb_q_down)

        word_atten_output_down = F.normalize(word_atten_output_down, dim=-1)

        with torch.no_grad():
            atten_weights_down = word_attn_q.detach()[:, ::2]
            word_atten_weights_down = []
            for i in range(bz):
                atten_weight = atten_weights_down[i]
                nonzero = atten_weight.nonzero().squeeze()
                if nonzero.numel() > 0:
                    low = torch.quantile(atten_weight[nonzero], 0.1)
                    high = torch.quantile(atten_weight[nonzero], 0.9)
                    atten_weight_clipped = atten_weight.clone()
                    atten_weight_clipped[nonzero] = atten_weight[nonzero].clip(low, high)
                else:
                    atten_weight_clipped = atten_weight.clone()
                word_atten_weights_down.append(atten_weight_clipped)
            word_atten_weights_down = torch.stack(word_atten_weights_down)

        word_atten_weights_down /= word_atten_weights_down.sum(dim=1, keepdims=True)

        word_sim_down = torch.bmm(word_emb_q_down, word_atten_output_down.permute(
            0, 2, 1)) / self.hparams.local_temperature
        word_num_down = word_sim_down.size(1)

        word_sim_1_down = rearrange(word_sim_down, "b n1 n2 -> (b n1) n2")
        targets_down = torch.arange(word_num_down).type_as(word_emb_q_down).long().repeat(bz)
        loss_sentence_1 = torch.sum(F.cross_entropy(
            word_sim_1_down, targets_down, reduction="none") * word_atten_weights_down.view(-1)) / bz

        word_sim_2_down = rearrange(word_sim_down, "b n1 n2 -> (b n2) n1")
        loss_sentence_2 = torch.sum(F.cross_entropy(
            word_sim_2_down, targets_down, reduction="none") * word_atten_weights_down.view(-1)) / bz

        loss_sentence = (loss_sentence_1 + loss_sentence_2) / 2.

        loss_local = loss_local + loss_sentence

        # normalize prototype layer
        with torch.no_grad():
            w = self.prototype_layer.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototype_layer.weight.copy_(w)

        # Compute assign code of images
        img_proto_out = self.prototype_layer(img_emb_q)
        report_proto_out = self.prototype_layer(report_emb_q)

        with torch.no_grad():
            img_code = torch.exp(
                img_proto_out / self.hparams.epsilon).t()
            img_code = self.get_assignments(
                img_code, self.hparams.sinkhorn_iterations)  # bz, 500
            report_code = torch.exp(
                report_proto_out / self.hparams.epsilon).t()
            report_code = self.get_assignments(
                report_code, self.hparams.sinkhorn_iterations)  # bz, 500

        img_proto_prob = F.softmax(
            img_proto_out / self.hparams.proto_temperature, dim=1)
        report_proto_prob = F.softmax(
            report_proto_out / self.hparams.proto_temperature, dim=1)

        loss_i2t_proto = - \
            torch.mean(
                torch.sum(img_code * torch.log(report_proto_prob), dim=1))
        loss_t2i_proto = - \
            torch.mean(torch.sum(report_code *
                                 torch.log(img_proto_prob), dim=1))

        loss_proto = (loss_i2t_proto + loss_t2i_proto) / 2.

        return loss_ita, loss_local, loss_proto, acc1, acc5

    def ssl_forward(self, batch):
        # ###### CL in image ######
        # Forward of query aug1 encoder
        aug1_feat_q, aug1_patch_feat_q = self.img_encoder_q(
            batch["augs1"])
        aug1_patch_emb_q = self.img_encoder_q.local_embed(aug1_patch_feat_q)
        aug1_patch_emb_q = F.normalize(aug1_patch_emb_q, dim=-1)
        aug1_emb_q = self.img_encoder_q.global_embed(aug1_feat_q)
        aug1_emb_q = F.normalize(aug1_emb_q, dim=-1)

        # Forward of query qug2 encoder
        aug2_feat_q, aug2_patch_feat_q = self.img_encoder_q(
            batch["augs2"])
        aug2_patch_emb_q = self.img_encoder_q.local_embed(aug2_patch_feat_q)
        aug2_patch_emb_q = F.normalize(aug2_patch_emb_q, dim=-1)
        aug2_emb_q = self.img_encoder_q.global_embed(aug2_feat_q)
        aug2_emb_q = F.normalize(aug2_emb_q, dim=-1)

        bz = aug1_emb_q.size(0)
        labels = torch.arange(bz).type_as(aug2_emb_q).long()

        scores = aug1_emb_q.mm(aug2_emb_q.t())
        scores /= self.hparams.softmax_temperature
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores1, labels)
        loss_aug_global = loss0 + loss1

        ########### Patch_level alignment ################
        # cross attention patch to sentences
        atten_sim = torch.bmm(aug1_patch_emb_q, aug2_patch_emb_q.permute(0, 2, 1))
        atten_scores = F.softmax(
            atten_sim / self.hparams.local_temperature, dim=-1)  # bz, num_patches, num_patches
        patch_atten_output_q1 = torch.bmm(atten_scores, aug2_patch_emb_q)

        patch_atten_output_q1 = F.normalize(patch_atten_output_q1, dim=-1)

        with torch.no_grad():
            patch_num = patch_atten_output_q1.size(1)
            patch_atten_weights = torch.ones(
                bz, patch_num).type_as(batch["augs1"]) / patch_num

        patch_sim = torch.bmm(aug1_patch_emb_q, patch_atten_output_q1.permute(
            0, 2, 1)) / self.hparams.local_temperature
        patch_num = patch_sim.size(1)
        patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
        targets = torch.arange(patch_num).type_as(
            aug1_patch_emb_q).long().repeat(bz)
        loss_patch_1 = torch.sum(F.cross_entropy(
            patch_sim_1, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

        patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
        loss_patch_2 = torch.sum(F.cross_entropy(
            patch_sim_2, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

        loss_patch = (loss_patch_1 + loss_patch_2) / 2.

        # Region Level
        aug1_patch_emb_q_down = aug1_patch_emb_q[:, ::2, :]  # [bz, num_patches//2, dim]
        aug2_patch_emb_q_down = aug2_patch_emb_q[:, ::2, :]  # [bz, num_patches//2, dim]

        atten_sim_down = torch.bmm(
            aug1_patch_emb_q_down,
            aug2_patch_emb_q_down.permute(0, 2, 1)
        )
        atten_scores_down = F.softmax(
            atten_sim_down / self.hparams.local_temperature,
            dim=-1
        )
        patch_atten_output_q1_down = torch.bmm(atten_scores_down, aug2_patch_emb_q_down)
        patch_atten_output_q1_down = F.normalize(patch_atten_output_q1_down, dim=-1)

        with torch.no_grad():
            patch_num_down = patch_atten_output_q1_down.size(1)
            patch_atten_weights_down = torch.ones(
                bz, patch_num_down
            ).type_as(batch["augs1"]) / patch_num_down

        patch_sim_down = torch.bmm(
            aug1_patch_emb_q_down,
            patch_atten_output_q1_down.permute(0, 2, 1)
        ) / self.hparams.local_temperature

        patch_sim_1_down = rearrange(patch_sim_down, "b n1 n2 -> (b n1) n2")
        targets_down = torch.arange(patch_num_down).type_as(aug1_patch_emb_q_down).long().repeat(bz)
        loss_region_1 = torch.sum(F.cross_entropy(
            patch_sim_1_down,
            targets_down,
            reduction="none"
        ) * patch_atten_weights_down.view(-1)) / bz

        patch_sim_2_down = rearrange(patch_sim_down, "b n1 n2 -> (b n2) n1")
        loss_region_2 = torch.sum(F.cross_entropy(
            patch_sim_2_down,
            targets_down,
            reduction="none"
        ) * patch_atten_weights_down.view(-1)) / bz

        loss_region = (loss_region_1 + loss_region_2) / 2.

        loss_patch = loss_patch + loss_region

        # Compute assign code of images
        aug1_proto_out = self.prototype_layer(aug1_emb_q)
        aug2_proto_out = self.prototype_layer(aug2_emb_q)

        with torch.no_grad():
            aug1_code = torch.exp(
                aug1_proto_out / self.hparams.epsilon).t()
            aug1_code = self.get_assignments(
                aug1_code, self.hparams.sinkhorn_iterations)  # bz, 500
            aug2_code = torch.exp(
                aug2_proto_out / self.hparams.epsilon).t()
            aug2_code = self.get_assignments(
                aug2_code, self.hparams.sinkhorn_iterations)  # bz, 500

        aug1_proto_prob = F.softmax(
            aug1_proto_out / self.hparams.proto_temperature, dim=1)
        aug2_proto_prob = F.softmax(
            aug2_proto_out / self.hparams.proto_temperature, dim=1)

        loss_a12a2_proto = - \
            torch.mean(
                torch.sum(aug1_code * torch.log(aug2_proto_prob), dim=1))
        loss_a22a1_proto = - \
            torch.mean(torch.sum(aug2_code *
                                 torch.log(aug1_proto_prob), dim=1))

        loss_aug_proto = (loss_a12a2_proto + loss_a22a1_proto) / 2.

        return loss_aug_global, loss_patch, loss_aug_proto

    def sinkhorn(self, Q, nmb_iters):
        '''
            :param Q: (num_prototypes, batch size)

        '''
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.hparams.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.hparams.gpus > 0:
                u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).cuda(
                    non_blocking=True) / (self.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def training_step(self, batch, batch_idx):
        (loss_ita, loss_local, loss_proto, loss_aug_global,
         loss_patch, loss_aug_proto, acc1, acc5) = self(
            batch, batch_idx, "train")
        loss = (self.hparams.lambda_1 * loss_ita + self.hparams.lambda_2 *
                loss_local + self.hparams.lambda_3 * loss_proto +
                self.hparams.lambda_1 * loss_aug_global + self.hparams.lambda_2 * loss_patch
                + self.hparams.lambda_3 * loss_aug_proto)

        log = {
            "train_loss": loss,
            "train_loss_ita": self.hparams.lambda_1 * loss_ita,
            "train_loss_local": self.hparams.lambda_2 * loss_local,
            "train_loss_proto": self.hparams.lambda_3 * loss_proto,
            "train_loss_aug_global": self.hparams.lambda_1 * loss_aug_global,
            "train_loss_patch": self.hparams.lambda_2 * loss_patch,
            "train_loss_aug_proto": self.hparams.lambda_3 * loss_aug_proto,
            "train_acc1": acc1,
            "train_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)

        return loss

    # freeze prototype layer
    def on_after_backward(self):
        if self.current_epoch < self.hparams.freeze_prototypes_epochs:
            for param in self.prototype_layer.parameters():
                param.grad = None

    def validation_step(self, batch, batch_idx):
        (loss_ita, loss_local, loss_proto, loss_aug_global,
         loss_patch, loss_aug_proto, acc1, acc5) = self(
            batch, batch_idx, "train")

        loss = (self.hparams.lambda_1 * loss_ita + self.hparams.lambda_2 *
                loss_local + self.hparams.lambda_3 * loss_proto +
                self.hparams.lambda_1 * loss_aug_global + self.hparams.lambda_2 * loss_patch
                + self.hparams.lambda_3 * loss_aug_proto)

        log = {
            "val_loss": loss,
            "val_loss_ita": self.hparams.lambda_1 * loss_ita,
            "val_loss_local": self.hparams.lambda_2 * loss_local,
            "val_loss_proto": self.hparams.lambda_3 * loss_proto,
            "val_loss_aug_global": self.hparams.lambda_1 * loss_aug_global,
            "val_loss_patch": self.hparams.lambda_2 * loss_patch,
            "val_loss_aug_proto": self.hparams.lambda_3 * loss_aug_proto,
            "val_acc1": acc1,
            "val_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=1e-8,
            warmup_steps=int(self.training_steps * 0.4)
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--img_encoder", type=str, default="vit_base")
        parser.add_argument("--freeze_bert", action="store_true")
        parser.add_argument("--emb_dim", type=int,
                            default=512, help="512, 762, 1024")
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--num_prototypes", type=int, default=512)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--lambda_1", type=float, default=1.5)
        parser.add_argument("--lambda_2", type=float, default=1.)
        parser.add_argument("--lambda_3", type=float, default=1.)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--bidirectional", action="store_false")
        parser.add_argument("--data_pct", type=float, default=1.)
        return parser

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs


@torch.no_grad()
def concat_all_gather(tensor):
    '''
    Performs all_gather operation on the provided tensors
    '''
    tensors_gather = [torch.ones_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    from datasets.datamodule import CLIP_DataModule, PIVLM_DataModule
    from datasets.transforms import DataTransforms, SSLDataTransforms

    parser = ArgumentParser()

    parser = PIVLM.add_model_specific_args(parser)
    args = parser.parse_args()

    args.deterministic = True
    args.max_epochs = 50

    datamodule = PIVLM_DataModule(CLIP_SSL_Dataset, DataTransforms,
                                  SSLDataTransforms, clip_ssl_collate_fn, 1.,
                                  16, 0)
    model = PIVLM(**args.__dict__)
    for b in datamodule.train_dataloader():
        a = b
        c = model(a, 1)
        d = 1
