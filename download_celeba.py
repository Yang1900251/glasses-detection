from pathlib import Path
import argparse
import json
import shutil
import random

import pandas as pd
from tqdm import tqdm


def find_celeba_root(raw_root: Path) -> Path:
    candidates = [
        raw_root,
        raw_root / "celeba-dataset",
        Path("data/data/raw_kaggle/celeba-dataset"),
        Path("data/raw_kaggle/celeba-dataset"),
        Path("data/data/raw_kaggle"),
        Path("data/raw_kaggle"),
    ]

    for root in candidates:
        if (root / "list_attr_celeba.csv").exists():
            return root

    data_dir = Path("data")
    if data_dir.exists():
        for attr_path in data_dir.rglob("list_attr_celeba.csv"):
            return attr_path.parent

    raise FileNotFoundError(
        "找不到 list_attr_celeba.csv。请检查 CelebA 数据集是否已经解压。"
    )


def find_image_dir(celeba_root: Path) -> Path:
    candidates = [
        celeba_root / "img_align_celeba" / "img_align_celeba",
        celeba_root / "img_align_celeba",
    ]

    for image_dir in candidates:
        if image_dir.exists() and any(image_dir.glob("*.jpg")):
            return image_dir

    target = list(celeba_root.rglob("000001.jpg"))
    if target:
        return target[0].parent

    raise FileNotFoundError(
        f"找不到 CelebA 图片目录，请检查路径：{celeba_root}"
    )


def normalize_attr_dataframe(attr_path: Path) -> pd.DataFrame:
    df = pd.read_csv(attr_path)

    if "image_id" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "image_id"})

    if "Eyeglasses" not in df.columns:
        raise ValueError(
            "list_attr_celeba.csv 中找不到 Eyeglasses 列，无法生成眼镜二分类标签。"
        )

    df = df[["image_id", "Eyeglasses"]].copy()
    df["image_id"] = df["image_id"].astype(str)
    df["label"] = (df["Eyeglasses"] == 1).astype(int)

    return df[["image_id", "label"]]


def stratified_sample(df: pd.DataFrame, mini_size: int, seed: int, balance: bool) -> pd.DataFrame:
    if mini_size <= 0 or mini_size >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if not balance:
        return df.sample(n=mini_size, random_state=seed).reset_index(drop=True)

    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0]

    n_each = mini_size // 2
    n_pos = min(n_each, len(pos_df))
    n_neg = min(n_each, len(neg_df))

    sampled_pos = pos_df.sample(n=n_pos, random_state=seed)
    sampled_neg = neg_df.sample(n=n_neg, random_state=seed + 1)

    selected = pd.concat([sampled_pos, sampled_neg], axis=0)

    remaining = mini_size - len(selected)
    if remaining > 0:
        used_ids = set(selected["image_id"])
        rest_df = df[~df["image_id"].isin(used_ids)]
        if len(rest_df) > 0:
            extra = rest_df.sample(
                n=min(remaining, len(rest_df)),
                random_state=seed + 2
            )
            selected = pd.concat([selected, extra], axis=0)

    return selected.sample(frac=1, random_state=seed + 3).reset_index(drop=True)


def add_split_column(df: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float) -> pd.DataFrame:
    random.seed(seed)

    result_parts = []

    for label_value in sorted(df["label"].unique()):
        part = df[df["label"] == label_value].sample(frac=1, random_state=seed + label_value)
        n = len(part)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_part = part.iloc[:n_train].copy()
        val_part = part.iloc[n_train:n_train + n_val].copy()
        test_part = part.iloc[n_train + n_val:].copy()

        train_part["split"] = "train"
        val_part["split"] = "val"
        test_part["split"] = "test"

        result_parts.extend([train_part, val_part, test_part])

    result = pd.concat(result_parts, axis=0)
    result = result.sample(frac=1, random_state=seed + 10).reset_index(drop=True)

    return result[["image_id", "label", "split"]]


def copy_images(df: pd.DataFrame, image_dir: Path, out_image_dir: Path) -> None:
    out_image_dir.mkdir(parents=True, exist_ok=True)

    missing = []

    for image_id in tqdm(df["image_id"].tolist(), desc="复制图片"):
        src = image_dir / image_id
        dst = out_image_dir / image_id

        if not src.exists():
            missing.append(image_id)
            continue

        if not dst.exists():
            shutil.copy2(src, dst)

    if missing:
        print(f"警告：有 {len(missing)} 张图片没有找到，示例：{missing[:5]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_root",
        type=str,
        default="data/data/raw_kaggle/celeba-dataset",
        help="CelebA 原始数据目录"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/mini_celeba",
        help="整理后的数据输出目录"
    )
    parser.add_argument(
        "--mini_size",
        type=int,
        default=5000,
        help="小数据集总图片数；设为 0 表示使用全部数据"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--no_balance",
        action="store_true",
        help="不做类别均衡采样"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出目录已存在，则先删除再重新生成"
    )

    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_image_dir = out_dir / "images"

    celeba_root = find_celeba_root(raw_root)
    image_dir = find_image_dir(celeba_root)
    attr_path = celeba_root / "list_attr_celeba.csv"

    print(f"使用 CelebA 根目录：{celeba_root}")
    print(f"使用图片目录：{image_dir}")
    print(f"使用属性文件：{attr_path}")
    print(f"输出目录：{out_dir}")

    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_attr_dataframe(attr_path)

    df["exists"] = df["image_id"].apply(lambda x: (image_dir / x).exists())
    df = df[df["exists"]].drop(columns=["exists"]).reset_index(drop=True)

    print("原始可用图片数量：", len(df))
    print("原始类别统计：")
    print(df["label"].value_counts().rename(index={0: "no_glasses", 1: "glasses"}))

    selected = stratified_sample(
        df=df,
        mini_size=args.mini_size,
        seed=args.seed,
        balance=not args.no_balance
    )

    selected = add_split_column(
        df=selected,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    copy_images(selected, image_dir, out_image_dir)

    labels_path = out_dir / "labels.csv"
    selected.to_csv(labels_path, index=False, encoding="utf-8-sig")

    class_to_idx = {
        "no_glasses": 0,
        "glasses": 1
    }

    with open(out_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    summary = (
        selected
        .groupby(["split", "label"])
        .size()
        .reset_index(name="count")
    )
    summary["class_name"] = summary["label"].map({0: "no_glasses", 1: "glasses"})
    summary = summary[["split", "class_name", "label", "count"]]
    summary.to_csv(out_dir / "split_summary.csv", index=False, encoding="utf-8-sig")

    print("\n整理完成。")
    print(f"labels.csv：{labels_path}")
    print("\n划分统计：")
    print(summary)


if __name__ == "__main__":
    main()