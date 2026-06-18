import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from model import GlassesCNN


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--output", type=str, default="results/predict_result.csv")

    args = parser.parse_args()

    image_path = Path(args.image)
    ckpt_path = Path(args.ckpt)
    output_path = Path(args.output)

    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片：{image_path}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到模型文件：{ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_path, map_location=device)
    img_size = checkpoint.get("img_size", 128)

    model = GlassesCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = build_transform(img_size)

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(image_tensor)
        prob = torch.sigmoid(logit).item()

    label = 1 if prob >= 0.5 else 0
    result = "戴眼镜" if label == 1 else "不戴眼镜"

    print(f"图片：{image_path}")
    print(f"预测结果：{result}")
    print(f"戴眼镜概率：{prob:.6f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {
                "image_path": str(image_path),
                "prob_glasses": prob,
                "predict_label": label,
                "predict_result": result,
            }
        ]
    )

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"预测结果已保存：{output_path}")


if __name__ == "__main__":
    main()