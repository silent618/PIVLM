import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
from transformers import BertTokenizer
from copy import deepcopy
import random
from .constants import *
from .utils import get_imgs


class CLIPDataset(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 imsize=256, max_words=224):
        super().__init__()
        if not os.path.exists(Quilt_data_dir):
            raise RuntimeError(f"{Quilt_data_dir} does not exist!")

        self.transform = transform
        self.imsize = imsize
        self.df = pd.read_csv(Quilt_csv_path)

        self.df = self.df[self.df['split'] == split]
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)

        self.tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = max_words

    def __len__(self):
        return len(self.df)

    def random_mask(self, tokens):
        masked_tokens = deepcopy(tokens)
        for i in range(1, masked_tokens.shape[1]-1):
            if masked_tokens[0][i] == 0:
                break

            prob = random.random()
            if prob < 0.5:
                masked_tokens[0][i] = 103

        return masked_tokens

    def get_caption(self, series_sent):
        if len(series_sent) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sent = list(filter(lambda x: x != "", series_sent))
        sent = " ".join(series_sent)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        """masked_ids = self.random_mask(tokens['input_ids'])
        tokens['masked_ids'] = masked_ids"""

        return tokens, x_len

    def __getitem__(self, index):
        key = Quilt_data_dir + self.df.iloc[index]['image_path']
        text = self.df.iloc[index]['caption']
        caps, cap_len = self.get_caption(text)
        image = get_imgs(key, self.imsize)
        if self.transform:
            img = self.transform(image)
        else:
            img = image
        return img, caps, cap_len, key


class CLIP_SSL_Dataset(data.Dataset):
    def __init__(self, split="train", transform=None, augment=None, data_pct=1.0,
                 imsize=256, max_words=224):
        super().__init__()
        if not os.path.exists(Quilt_data_dir):
            raise RuntimeError(f"{Quilt_data_dir} does not exist!")

        self.transform = transform
        self.augment = augment
        self.imsize = imsize
        self.df = pd.read_csv(Quilt_csv_path)

        self.df = self.df[self.df['split'] == split]
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)

        self.tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = max_words

    def __len__(self):
        return len(self.df)

    def random_mask(self, tokens):
        masked_tokens = deepcopy(tokens)
        for i in range(1, masked_tokens.shape[1]-1):
            if masked_tokens[0][i] == 0:
                break

            prob = random.random()
            if prob < 0.5:
                masked_tokens[0][i] = 103

        return masked_tokens

    def get_caption(self, series_sent):
        if len(series_sent) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sent = list(filter(lambda x: x != "", series_sent))
        sent = " ".join(series_sent)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        """masked_ids = self.random_mask(tokens['input_ids'])
        tokens['masked_ids'] = masked_ids"""

        return tokens, x_len

    def __getitem__(self, index):
        key = Quilt_data_dir + self.df.iloc[index]['image_path']
        text = self.df.iloc[index]['caption']
        caps, cap_len = self.get_caption(text)
        image = get_imgs(key, self.imsize)
        if self.transform:
            img = self.transform(image)
        else:
            img = image
        aug1 = self.augment(image)
        aug2 = self.augment(image)
        return img, aug1, aug2, caps, cap_len, key


def clip_collate_fn(batch):
    """sort sequence"""
    imgs, cap_len, ids, tokens, attention = [], [], [], [], []
    path = []
    for b in batch:
        img, cap, cap_l, p = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)

    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(
        torch.tensor(cap_len), 0, True)

    path = np.array(path)

    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path[sorted_cap_indices],
    }
    return return_dict


def clip_ssl_collate_fn(batch):
    """sort sequence"""
    imgs, augs1, augs2, cap_len, ids, tokens, attention = [], [], [], [], [], [], []
    path = []
    for b in batch:
        img, aug1, aug2, cap, cap_l, p = b
        imgs.append(img)
        augs1.append(aug1)
        augs2.append(aug2)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)

    # stack
    imgs = torch.stack(imgs)
    augs1 = torch.stack(augs1)
    augs2 = torch.stack(augs2)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(
        torch.tensor(cap_len), 0, True)

    path = np.array(path)

    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "augs1": augs1[sorted_cap_indices],
        "augs2": augs2[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path[sorted_cap_indices],
    }
    return return_dict


if __name__ == '__main__':
    from datamodule import CLIP_DataModule, PIVLM_DataModule
    from transforms import DataTransforms, SSLDataTransforms
    datamodule = PIVLM_DataModule(CLIP_SSL_Dataset, DataTransforms,
                                  SSLDataTransforms, clip_ssl_collate_fn, 1.,
                                  16, 0)
    for b in datamodule.train_dataloader():
        a = 1
    """dataset = MultimodalPretrainingDataset(split="train")
    data = dataset[0][0]"""
    """df = pd.read_csv("F:/Quilt/quilt_1M_lookup.csv")
    df = df[df['split'] == 'val']"""
    a = 1
