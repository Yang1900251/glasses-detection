from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class GlassesDataset(Dataset):
    def __init__(self, data_dir="data/mini_celeba", split="train", transform=None):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.label_path = self.data_dir / "labels.csv"
        self.split = split
        self.transform = transform

        if not self.image_dir.exists():
            raise FileNotFoundError(f"找不到图片文件夹：{self.image_dir}")

        if not self.label_path.exists():
            raise FileNotFoundError(f"找不到标签文件：{self.label_path}")

        df = pd.read_csv(self.label_path)

        required_columns = {"image_id", "label", "split"}
        if not required_columns.issubset(set(df.columns)):
            raise ValueError("labels.csv 必须包含 image_id、label、split 三列")

        if split not in {"train", "val", "test"}:
            raise ValueError("split 只能是 train、val 或 test")

        df = df[df["split"] == split].copy().reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"{split} 集为空，请重新检查 labels.csv")

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_id = str(row["image_id"])
        label = int(row["label"])

        image_path = self.image_dir / image_id

        if not image_path.exists():
            raise FileNotFoundError(f"找不到图片：{image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


def get_class_counts(dataset: GlassesDataset):
    counts = dataset.df["label"].value_counts().to_dict()
    return {
        0: int(counts.get(0, 0)),
        1: int(counts.get(1, 0))
    }