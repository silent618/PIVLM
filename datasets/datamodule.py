import pytorch_lightning as pl
from torch.utils.data import DataLoader


class CLIP_DataModule(pl.LightningDataModule):
    def __init__(self, dataset, transforms, collate_fn, data_pct, batch_size, num_workers, crop_size=224):
        super().__init__()

        self.dataset = dataset
        self.transforms = transforms
        self.collate_fn = collate_fn
        self.data_pct = data_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size

    def train_dataloader(self):
        if self.transforms:
            transform = self.transforms(True, self.crop_size)
        else:
            transform = None

        dataset = self.dataset(
            split="train", transform=transform, data_pct=self.data_pct)

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            split="valid", transform=transform, data_pct=self.data_pct)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )


class PIVLM_DataModule(pl.LightningDataModule):
    def __init__(self, dataset, transforms, augment, collate_fn, data_pct, batch_size, num_workers, crop_size=224):
        super().__init__()

        self.dataset = dataset
        self.transforms = transforms
        self.collate_fn = collate_fn
        self.augment = augment()
        self.data_pct = data_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size

    def train_dataloader(self):
        if self.transforms:
            transform = self.transforms(True, self.crop_size)
        else:
            transform = None

        dataset = self.dataset(
            split="train", transform=transform, augment=self.augment, data_pct=self.data_pct)

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            split="val", transform=transform, augment=self.augment, data_pct=self.data_pct)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )


# if __name__=="__main__":
